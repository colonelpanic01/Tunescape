import os
import spotipy
from flask import Flask, redirect, request, session, render_template
from spotipy.oauth2 import SpotifyOAuth
# import base64
# from huggingface_hub import InferenceClient
import requests

import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import load_image, export_to_video

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Spotify Configuration
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:5000/callback")
SCOPE = 'user-library-read playlist-read-private'

# Runway ML Configuration
# RUNWAY_API_KEY = os.getenv("RUNWAYML_API_KEY")

# Initialize Spotify OAuth
sp_oauth = SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=SCOPE
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
def generate_video_huggingface(image_path, prompt):
    """
    Generate video using Hugging Face Inference API
    
    Args:
        image_path (str): Path to the source image
        prompt (str): Descriptive prompt for video generation
    
    Returns:
        str: Path to generated video file
    """
    # Initialize Hugging Face Inference Client
    # client = InferenceClient(
    #     token=os.getenv('HUGGINGFACE_API_TOKEN'),
    #     model="cerspense/zeroscope_v2_576w"
    # )
    
    pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float32)
    
    # load weights from local directory 
    # pipeline = I2VGenXLPipeline.from_pretrained("./models/i2vgen-xl", torch_dtype=torch.float32)

    # Read and encode image
    # with open(image_path, "rb") as image_file:
    #     base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    # image = load_image(base64_image).convert("RGB")
    image = load_image(image_path).convert("RGB")
    # prompt = "Papers were floating in the air on a table in the library"
    # negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
    generator = torch.manual_seed(8888)
    # Prepare API payload
    frames = pipeline(
        prompt=prompt,
        image=image,
        # num_inference_steps=100,
        num_inference_steps = 50,
        # negative_prompt=negative_prompt,
        guidance_scale=9.0,
        generator=generator
    ).frames[0]

    # payload = {
    #     "inputs": prompt,
    #     "image": f"data:image/png;base64,{base64_image}",
    #     "parameters": {
    #         "num_inference_steps": 50,
    #         "num_frames": 24,  # 1 second at 24 fps
    #         "width": 256,
    #         "height": 256
    #     }
    # }
    
    try:
        # Generate video
        # video_bytes = client.image_to_video(payload)
        # video_bytes = client.export_to_video(payload)
        video_bytes = export_to_video(frames, 'output_video.mp4', fps=10)
        
        # Save video
        output_path = os.path.join('static', 'generated_videos', 'output_video.mp4')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(video_bytes)
        
        return output_path
    
    except Exception as e:
        print(f"Video generation error: {e}")
        return None


@app.route('/generate_video', methods=['POST'])
def generate_video():
    sp = get_spotify_client()
    if not sp:
        return redirect('/')
    
    song_id = request.form.get('song_id')
    art_style = request.form.get('art_style', 'cinematic')
    
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
    
    # Track details for context
    track = sp.track(song_id)
    track_name = track['name']
    artist_name = track['artists'][0]['name']
    
    # Generate video prompt
    video_prompt = (
        f"Transform the album art of {track_name} by {artist_name} "
        f"into a {art_style} style video. "
        "Smooth, cinematic movement revealing musical essence."
    )
    
    # Generate video
    video_path = generate_video_huggingface(local_image_path, video_prompt)
    
    if video_path:
        return render_template('video_result.html', 
                               video_url=video_path, 
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