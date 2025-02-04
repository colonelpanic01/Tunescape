import os
import time
import base64
import spotipy
from flask import Flask, redirect, request, session, render_template
from spotipy.oauth2 import SpotifyOAuth
from runwayml import RunwayML
import requests

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Spotify Configuration
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:5000/callback")
SCOPE = 'user-library-read playlist-read-private'

# Runway ML Configuration
RUNWAY_API_KEY = os.getenv("RUNWAYML_API_KEY")

# Initialize Spotify OAuth
sp_oauth = SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=SCOPE
)

def get_spotify_client():
    """Helper function to get Spotify client with refreshed token"""
    token_info = session.get('token_info', None)
    if not token_info:
        return None
    
    # Refresh token if expired
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        session['token_info'] = token_info
    
    return spotipy.Spotify(auth=token_info['access_token'])

def get_album_cover(sp, track_id):
    """
    Retrieve the highest resolution album cover for a track
    
    Args:
        sp (spotipy.Spotify): Authenticated Spotify client
        track_id (str): Spotify track ID
    
    Returns:
        str: URL of the album cover image
    """
    track = sp.track(track_id)
    
    # Get album images sorted by size (largest first)
    album_images = track['album']['images']
    if album_images:
        # Select the largest image (first in the list)
        cover_url = album_images[0]['url']
        return cover_url
    
    return None

def download_image(image_url, filename='album_cover.png'):
    """
    Download image from URL and save locally
    
    Args:
        image_url (str): URL of the image to download
        filename (str): Local filename to save
    
    Returns:
        str: Path to the downloaded image
    """
    response = requests.get(image_url)
    if response.status_code == 200:
        filepath = os.path.join('static', 'album_covers', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return filepath
    
    return None

@app.route('/')
def index():
    """Render login page or redirect to dashboard if logged in"""
    if 'token_info' not in session:
        return render_template('login.html')
    return redirect('/dashboard')

@app.route('/login')
def login():
    """Redirect to Spotify login"""
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    """Handle Spotify OAuth callback"""
    token_info = sp_oauth.get_access_token(request.args['code'])
    session['token_info'] = token_info
    return redirect('/dashboard')

@app.route('/dashboard')
def dashboard():
    """User dashboard to select songs and generate art/video"""
    sp = get_spotify_client()
    if not sp:
        return redirect('/')
    
    # Fetch user's playlists with tracks
    playlists_data = []
    playlists = sp.current_user_playlists(limit=50)
    
    for playlist in playlists['items']:
        # Fetch tracks for each playlist
        playlist_tracks = sp.playlist_tracks(playlist['id'])
        tracks = []
        for item in playlist_tracks['items']:
            track = item.get('track', {})
            if track:
                tracks.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artists': track['artists'],
                    'album_cover': track['album']['images'][0]['url'] if track['album']['images'] else None
                })
        
        playlists_data.append({
            'name': playlist['name'],
            'tracks': tracks
        })
    
    # Fetch liked songs
    liked_songs = []
    saved_tracks = sp.current_user_saved_tracks(limit=50)
    for item in saved_tracks['items']:
        track = item.get('track', {})
        if track:
            liked_songs.append({
                'id': track['id'],
                'name': track['name'],
                'artists': track['artists'],
                'album_cover': track['album']['images'][0]['url'] if track['album']['images'] else None
            })
    
    return render_template('dashboard.html', 
                           playlists=playlists_data, 
                           liked_songs=liked_songs)

@app.route('/generate_video', methods=['POST'])
def generate_video():
    """Generate video from album cover using Runway ML"""
    sp = get_spotify_client()
    if not sp:
        return redirect('/')
    
    song_id = request.form.get('song_id')
    art_style = request.form.get('art_style', 'cinematic')
    
    # Initialize Runway ML Client
    runway_client = RunwayML(api_key=RUNWAY_API_KEY)
    
    # Retrieve album cover
    album_cover_url = get_album_cover(sp, song_id)
    
    if not album_cover_url:
        return render_template('error.html', 
                               error='Could not retrieve album cover')
    
    # Download album cover
    local_image_path = download_image(album_cover_url)
    
    if not local_image_path:
        return render_template('error.html', 
                               error='Failed to download album cover')
    
    # Encode image to base64
    with open(local_image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    # Track details for context
    track = sp.track(song_id)
    track_name = track['name']
    artist_name = track['artists'][0]['name']
    
    # Create image-to-video task
    task = runway_client.image_to_video.create(
        model='gen3a_turbo',
        prompt_image=f"data:image/png;base64,{base64_image}",
        prompt_text=f'Transform {track_name} album art by {artist_name} into a {art_style} video'
    )
    
    task_id = task.id
    
    # Poll task status
    time.sleep(10)  # Initial wait
    task = runway_client.tasks.retrieve(task_id)
    
    while task.status not in ['SUCCEEDED', 'FAILED']:
        time.sleep(10)  # Wait between polls
        task = runway_client.tasks.retrieve(task_id)
    
    if task.status == 'SUCCEEDED':
        # Render video result page
        return render_template('video_result.html', 
                               video_url=task.output.video_url, 
                               song_name=track_name, 
                               artist_name=artist_name,
                               album_cover_url=album_cover_url)
    else:
        return render_template('error.html', error='Video generation failed')

@app.route('/logout')
def logout():
    """Clear user session"""
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)