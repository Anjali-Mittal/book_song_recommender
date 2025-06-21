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
# Get song recs using Spotify API
def get_spotify_recommendations(user_query, tone=None, limit=10):
    try:
        search_results = sp.search(q=user_query, type='track', limit=1)
        if not search_results['tracks']['items']:
            return []

        seed_track = search_results['tracks']['items'][0]
        seed_track_id = seed_track['id']

        mood_valence = {
            "Joy": 0.9, "Surprise": 0.7, "Anger": 0.2,
            "Fear": 0.3, "Sadness": 0.1, "Trust": 0.6
        }
        valence = mood_valence.get(tone, 0.5)

        try:
            recs = sp.recommendations(
                seed_tracks=[seed_track_id],
                limit=limit,
                target_valence=valence
            )
            songs = []
            seen = set()  # To track unique songs

            for track in recs['tracks']:
                    title = track['name']
                    artist = track['artists'][0]['name']
                    key = (title.lower(), artist.lower())  # Case-insensitive

                    if key in seen or len(title) > 35:
                        continue  # Skip duplicates or long titles

                    seen.add(key)
                    songs.append({
                        'title': title,
                        'artist': artist,
                        'spotify_url': track['external_urls']['spotify'],
                        'image_url': track['album']['images'][0]['url'] if track['album']['images'] else ""
                    })
            return songs

        except Exception as e:
            print(f"[Spotify Fallback] Rec error: {e}")
            # fallback: search again with broader limit
            fallback_results = sp.search(q=user_query, type='track', limit=limit)
            songs = []
            for track in fallback_results['tracks']['items']:
                songs.append({
                    'title': track['name'],
                    'artist': track['artists'][0]['name'],
                    'spotify_url': track['external_urls']['spotify'],
                    'image_url': track['album']['images'][0]['url'] if track['album']['images'] else ""
                })

            return songs

    except Exception as e:
        print(f"[Spotify Error] {e}")
        return []



def semantic_recommendation(query,
            category: str='None',
            tone: str='No',
            initial_top_k:int=100,
            top_k:int=20)-> pd.DataFrame:
    recs=db_books.similarity_search(query,k=initial_top_k)
    books_list=[int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs=books[books["isbn13"].isin(books_list)].head(top_k)
    # if user chose a filter category, filter the recommendations
    if category != "All":
        book_recs = book_recs[book_recs["mapped_category"] == category][:top_k]
    else:
        book_recs = book_recs.head(top_k)

    # filter according to a particular tone
    if tone == "Joy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprise":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Anger":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Fear":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sadness":
        book_recs.sort_values(by="sadness",ascending=False,inplace=True)
    elif tone == "Trust":
        book_recs.sort_values(by="trust",ascending=False,inplace=True)
    return book_recs


def get_book_details(query: str, category: str, tone: str):
    try:
        recommendations = semantic_recommendation(query, category, tone)
        html_output = ""

        for _, row in recommendations.iterrows():
            authors_split = row["authors"].split(";")
            if len(authors_split) == 2:
                authors = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors = row["authors"]

            description = row["description"]
            truncated_description = (
                description[:300] + "..." if len(description) > 300 else description
            )

            html_output += f"""
            <div style="display:inline-block;width:200px;margin:10px;vertical-align:top;">
                <img src="{row['large_thumbnail']}" width="150" height="220" style="border-radius:8px;"><br>
                <b>{row['title']}</b><br><i>by {authors}</i><br><small>{truncated_description}</small>
            </div>
            """

        # 🎧 Add Spotify recommendations
        try:
            songs = get_spotify_recommendations(query, tone if tone != "All" else None)
            if songs:
                html_output += "<hr><b>🎵 Matching Songs</b><div style='overflow-x:auto; white-space:nowrap;'>"
                for song in songs:
                        if len(song['title']) > 35:
                            continue  # Skip long titles

                        html_output += f"""
                        <div style="display:inline-block; width:180px; margin:10px; vertical-align:top;">
                            <a href="{song['spotify_url']}" target="_blank" style="text-decoration:none; color:black;">
                                <div style="text-align:center;">
                                    <b style="display:block; margin-bottom:5px;">{song['title']}</b>
                                    <img src="{song.get('image_url', '')}" width="140" height="140" style="border-radius:8px; display:block; margin:auto;">
                                    <small style="display:block; margin-top:5px;">{song['artist']}</small>
                                </div>
                            </a>
                        </div>
                        """


                html_output += "</div>"
        except Exception as e:
            print(f"[Song Rec Error] {e}")

        # 🔧 If nothing collected, show fallback
        if not html_output:
            html_output = "No recommendations found."

        print(f"[DEBUG] Book recs: {len(recommendations)}")
        print(f"[DEBUG] Song recs: {len(songs) if 'songs' in locals() else 0}")
        return html_output

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

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
