import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="my_chromadb")

# Distilbert-base-nli-mean-tokens model for embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="distilbert-base-nli-mean-tokens")

# Get or create the collection
collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})

def main():
    st.title("Search Engine Relevance for Video Subtitles🔍")

    # Getting the user input
    user_query = st.text_input("Input a query to begin your search:")

    if st.button("Search"):
        if user_query:
            # Query the collection
            results = collection.query(
                query_texts=[user_query],
                n_results=10,
                include=['documents', 'distances', 'metadatas']
            )

            # Display user input
            st.write(f"Your search query: {user_query}")

            # Display output documents
            st.write("Search Results:")
            for i, document in enumerate(results['documents'][0], 1):
                st.write(f"{i}. {document}")

if __name__ == "_main_":
    main()