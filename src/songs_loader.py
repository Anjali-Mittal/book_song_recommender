import json
from langchain.docstore.document import Document

def load_song_documents(path):
    song_documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            track = json.loads(line.strip())
            content = f"{track['title']} by {track['artist']}"
            metadata = {
                "title": track["title"],
                "artist": track["artist"],
                "spotify_url": track["spotify_url"],
                "image_url": track["image_url"] if track.get("image_url", "").startswith("http") else None

            }
            song_documents.append(Document(page_content=content, metadata=metadata))
    return song_documents








# import os
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from dotenv import load_dotenv
# from spotipy.oauth2 import SpotifyClientCredentials
# import spotipy
# load_dotenv()
# model_name = os.getenv("EMBEDDING_MODEL")
# chroma_dir = os.getenv("CHROMA_DB_DIR")
# os.environ['SPOTIPY_CLIENT_ID'] = os.getenv("SPOTIPY_CLIENT_ID")
# os.environ['SPOTIPY_CLIENT_SECRET'] = os.getenv("SPOTIPY_CLIENT_SECRET")

# # Load using environment variable
# embedding_model = HuggingFaceEmbeddings(model_name=model_name)
# sp=spotipy.Spotify(auth_manager=SpotifyClientCredentials())

# db_songs = Chroma(
#     persist_directory="song_vector_db",
#     embedding_function=embedding_model
# )
# docs = db_songs.similarity_search("test", k=1)
# print(docs[0].metadata)

