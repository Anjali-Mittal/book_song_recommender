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
model_name = "sentence-transformers/all-MiniLM-L6-v2"
chroma_dir = os.getenv("CHROMA_DB_DIR")

# Load the Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

# Create Chroma vector DB
db_books = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=chroma_dir
)

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
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone == "Trust":
        book_recs.sort_values(by="trust", ascending=False, inplace=True)
    return book_recs


def get_book_details(query: str, category: str, tone: str):
    recommendations = semantic_recommendation(query, category, tone)
    html_output = ""

    for _, row in recommendations.iterrows():
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split)>2:
            authors=f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors=row["authors"]

        description=row["description"]
        truncated_description = (
            description[:300] + "..." if len(description)>300 else description
        )

        html_output += f"""
        <div style="display:inline-block;width:200px;margin:10px;vertical-align:top;">
            <img src="{row['large_thumbnail']}" width="150" height="220" style="border-radius:8px;"><br>
            <b>{row['title']}</b><br><i>by {authors}</i><br><small>{truncated_description}</small>
        </div>
        """

    return html_output



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

