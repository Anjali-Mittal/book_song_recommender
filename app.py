import pandas as pd
import numpy as np
import gradio as gr
import os
import sentence_transformers
# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '3' to suppress all, '2' to hide info and warnings
import tf_keras as keras
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.songs_loader import load_song_documents

load_dotenv()
books=pd.read_csv("books_with_emotions.csv")

#loading best pixel size for book covers
books["large_thumbnail"] = books["thumbnail"] +"&fife=w800"
books["large_thumbnail"]=np.where(books["large_thumbnail"].isna(),
                    "cover_not_available.png",
                    books["large_thumbnail"])


# Semantic Recomendations
# reading tagged descriptions splitting each line and applying to doc chunks and changing to vector database to retrive nearest one

raw_documents=TextLoader("tagged_description.txt",encoding="utf-8").load()
text_splitter=CharacterTextSplitter(chunk_size=0,chunk_overlap=0,separator="\n")
from langchain.docstore.document import Document
documents = [
    Document(
        page_content=str(row["tagged_description"]),
        metadata={"isbn13": row["isbn13"], "title": row["title"]}
    )
    for _, row in books.iterrows()
]
 #Loading env variables
model_name = os.getenv("EMBEDDING_MODEL")
chroma_dir = os.getenv("CHROMA_DB_DIR")
os.environ['SPOTIPY_CLIENT_ID'] = os.getenv("SPOTIPY_CLIENT_ID")
os.environ['SPOTIPY_CLIENT_SECRET'] = os.getenv("SPOTIPY_CLIENT_SECRET")

# Load using environment variable
embedding_model = HuggingFaceEmbeddings(model_name=model_name)
sp=spotipy.Spotify(auth_manager=SpotifyClientCredentials())

# Create Chroma vector DB
db_books = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=chroma_dir
)

db_songs = Chroma(
    persist_directory="song_vector_db",
    embedding_function=embedding_model
)

def get_spotify_recommendations(user_query, tone=None, limit=10):
    try:
        recs = db_songs.similarity_search(user_query, k=50)  # slightly higher to allow filtering
        seen = set()
        songs = []

        for rec in recs:
            meta = rec.metadata
            title = meta["title"]
            artist = meta["artist"]
            image_url = meta.get("image_url", "")
            key = (title.lower(), artist.lower())

            if key in seen or len(title) > 35:
                continue

            seen.add(key)
            songs.append({
                "title": title,
                "artist": artist,
                "spotify_url": meta["spotify_url"],
                "image_url": image_url
            })

            if len(songs) >= limit:
                break

        return songs

    except Exception as e:
        print(f"[Semantic Spotify Search Error] {e}")
        return []
    
# Semantic Book Recommendations
def semantic_recommendation(query,
            category: str='None',
            tone: str='No',
            top_k:int=30,
            initial_top_k:int=30*30,) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.metadata['isbn13']) for rec in recs if 'isbn13' in rec.metadata]

    book_recs = books[books["isbn13"].isin(books_list)]

    if category != "All":
        book_recs = book_recs[book_recs["mapped_category"] == category]

    if tone == "Joy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprise":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Anger":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Fear":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sadness":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone == "Trust":
        book_recs.sort_values(by="trust", ascending=False, inplace=True)

    return book_recs.head(top_k)




def get_book_details(query: str, category: str, tone: str):
    try:
        recommendations = semantic_recommendation(query, category, tone)
        html_output = ""

        # 📚 Book Recommendations
        if not recommendations.empty:
            html_output += "<b>📚 Recommended Books</b><br>"
            for _, row in recommendations.iterrows():
                authors_split = row["authors"].split(";")
                if len(authors_split) == 2:
                    authors = f"{authors_split[0]} and {authors_split[1]}"
                elif len(authors_split) > 2:
                    authors = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
                else:
                    authors = row["authors"]

                description = row.get("description", "")
                truncated_description = (
                    description[:300] + "..." if len(description) > 300 else description
                )
                image_url = row.get('large_thumbnail') or "cover_not_available.png"

                html_output += f"""
                <div style="display:inline-block;width:200px;margin:10px;vertical-align:top;">
                    <img src="{image_url}" width="150" height="220" style="border-radius:8px;"><br>
                    <b>{row['title']}</b><br><i>by {authors}</i><br><small>{truncated_description}</small>
                </div>
                """
        else:
            html_output += "<p>No books found matching your description.</p>"

        # 🎧 Spotify Song Recommendations
        try:
            songs = get_spotify_recommendations(query, tone if tone != "All" else None)
            if songs:
                html_output += "<hr><b>🎵 Matching Songs</b><div style='overflow-x:auto; white-space:nowrap;'>"
                for song in songs:
                    title = song.get('title', 'Untitled')
                    artist = song.get('artist', 'Unknown Artist')
                    image = song.get('image_url')
                    if not image or not image.startswith("http"):
                        image = "https://via.placeholder.com/140x140.png?text=No+Cover"

                    url = song.get('spotify_url', '#')

                    html_output += f"""
                    <div style="display:inline-block; width:180px; margin:10px; vertical-align:top;">
                        <a href="{url}" target="_blank" style="text-decoration:none; color:black;">
                            <div style="text-align:center;">
                                <b style="display:block; margin-bottom:5px;">{title}</b>
                                <img src="{image}" width="140" height="140" style="border-radius:8px; display:block; margin:auto;">
                                <small style="display:block; margin-top:5px;">{artist}</small>
                            </div>
                        </a>
                    </div>
                    """
                html_output += "</div>"
            else:
                html_output += "<p>No matching songs found.</p>"

        except Exception as song_error:
            print(f"[Song Rec Error] {song_error}")
            html_output += "<p>⚠️ Couldn't fetch songs right now. Try again later.</p>"

        return html_output

    except Exception as main_error:
        print(f"[Book Rec Error] {main_error}")
        return "<p>⚠️ Something went wrong while fetching your recommendations. Please try a different query.</p>"


# CREATING DROPDOWNS
categories =["All"] + sorted(books["mapped_category"].dropna().astype(str).unique().tolist())

tones=["All","Joy","Surprise","Anger","Fear","Sadness","Trust"]

#SETTING UP THEME AND DASHBOARd
with gr.Blocks(theme=gr.themes.Soft) as dashboard:
    gr.Markdown("## 📚Book Recommender\nDescribe what you’re feeling or looking for in a book.")

    with gr.Row():
            user_query = gr.Textbox(
            label="📖 Describe a book you want",
            placeholder="A thrilling mystery...",
        )

    with gr.Row():
        category_filter = gr.Dropdown(label="📂 Category", choices=categories, value="All")
        tone_filter = gr.Dropdown(label="🎭 Emotion", choices=tones, value="All")
        submit_button = gr.Button("🔍 Get Recommendations", variant="primary")

    gr.Markdown("---")
    gr.Markdown("### Recommendations")
    output_container = gr.HTML()


    # Connect the button to the custom UI output
    submit_button.click(
        fn=get_book_details,
        inputs=[user_query, category_filter, tone_filter],
        outputs=output_container,
    )

    if __name__ == "__main__":
        # Launch the Gradio app
        dashboard.launch(share=True)