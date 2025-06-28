import json
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from artist_whitelist import ARTIST_WHITELIST  # import your list
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["SPOTIPY_CLIENT_ID"] = os.getenv("SPOTIPY_CLIENT_ID")
os.environ["SPOTIPY_CLIENT_SECRET"] = os.getenv("SPOTIPY_CLIENT_SECRET")


sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

def get_artist_id(artist_name):
    try:
        results = sp.search(q=f"artist:{artist_name}", type='artist', limit=1)
        items = results.get('artists', {}).get('items', [])
        if items:
            return items[0]['id']
    except Exception as e:
        print(f"Error getting ID for {artist_name}: {e}")
    return None

def get_artist_tracks(artist_id):
    try:
        results = sp.artist_top_tracks(artist_id)
        return results['tracks']
    except Exception as e:
        print(f"Error fetching tracks for artist {artist_id}: {e}")
        return []

def save_tracks_to_jsonl(tracks, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        for track in tracks:
            data = {
                "title": track['name'],
                "artist": track['artists'][0]['name'],
                "spotify_url": track['external_urls']['spotify'],
                "image_url": track['album']['images'][0]['url'] if track['album']['images'] else ""
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def main():
    output_file = "spotify_tracks.jsonl"
    seen = set()
    for artist in ARTIST_WHITELIST:
        print(f"Processing: {artist}")
        artist_id = get_artist_id(artist)
        if not artist_id:
            continue
        tracks = get_artist_tracks(artist_id)
        new_tracks = []
        for track in tracks:
            key = (track['name'].lower(), track['artists'][0]['name'].lower())
            if key not in seen:
                seen.add(key)
                new_tracks.append(track)
        save_tracks_to_jsonl(new_tracks, output_file)
        time.sleep(1)  # avoid hitting rate limits

if __name__ == "__main__":
    main()
