import os
from flask import Flask, request, jsonify
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from chromadb import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv, find_dotenv
import openai
from sentence_transformers import CrossEncoder
import numpy as np
from logging.handlers import StreamHandler  

# Load environment variables
load_dotenv(find_dotenv())

# Set OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

# Initialize Flask application
app = Flask(__name__)

# Initialize ChromaDB client
chroma_client = Client()

def preprocess_document(docx_path):
    doc = Document(docx_path)
    docx_texts = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]

    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=500, chunk_overlap=0)
    character_split_texts = character_splitter.split_text('\n\n'.join(docx_texts))

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts

def create_chromadb_collection(collection_name, documents):
    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_collection = chroma_client.create_collection(collection_name, embedding_function=embedding_function)
    ids = [str(i) for i in range(len(documents))]
    chroma_collection.add(ids=ids, documents=documents)
    return chroma_collection

def query_documents(query, collection):
    results = collection.query(query_texts=[query], n_results=5)
    retrieved_documents = results['documents'][0]
    return retrieved_documents

def augment_multiple_query(query, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert contract advisor assistant. Your users are asking questions about information contained in the contract."
                       "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
                       "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
                       "Make sure they are complete questions, and that they are related to the original question."
                       "Output one question per line. Do not number the questions."
        },
        {"role": "user", "content": query}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message['content']
    content = content.split("\n")
    return content

def rag_with_query_expansion(original_query, collection, model="gpt-3.5-turbo"):
    # Expand the original query
    expanded_queries = augment_multiple_query(original_query, model=model)[:5]  # Limit to top 5 expanded queries

    # Include the original query in the list of queries
    queries = [original_query] + expanded_queries

    # Query ChromaDB for relevant documents
    results = collection.query(query_texts=queries, n_results=5, include=['documents'])
    retrieved_documents = results['documents']

    # Deduplicate the retrieved documents
    unique_documents = set()
    for documents in retrieved_documents:
        for document in documents:
            unique_documents.add(document)

    # Convert back to list for indexing
    unique_documents = list(unique_documents)

    # Rank documents based on relevance to each query
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, doc] for doc in unique_documents for query in queries]
    scores = cross_encoder.predict(pairs)

    # Sort documents based on average score across all queries
    avg_scores = np.mean(np.array(scores).reshape(len(queries), -1), axis=0)
    ranked_indices = np.argsort(avg_scores)[::-1]
    ranked_documents = [unique_documents[i] for i in ranked_indices]

    # Generate answer using RAG with ranked documents
    information = "\n\n".join(ranked_documents[:5])  # Use top 5 ranked documents
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert legal contract advisor assistant. Your users are asking questions about information contained in the contract."
                       "You will be shown the user's question, and the relevant information from the contract. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {original_query}. \n Information: {information}"},
        {
        "role": "system",
        "content": "Provide concise answers without additional explanation unless explicitly requested."
                    "Identify questions that require a simple yes or no response, and Provide concise answers without additional explanation unless explicitly requested."
        }
    ]
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message['content']
    return content

@app.route('/answer', methods=['POST'])
def get_answer():
    data = request.get_json()

    if 'docx_path' not in data or 'query' not in data:
        return jsonify({"error": "Missing 'docx_path' or 'query' in request body"}), 400

    docx_path = data['docx_path']
    original_query = data['query']

    # Process the document
    documents = preprocess_document(docx_path)

    # Create ChromaDB collection and add documents
    collection_name = "Contract"
    chroma_collection = create_chromadb_collection(collection_name, documents)

    # Generate answer with query expansion and RAG
    output = rag_with_query_expansion(original_query, chroma_collection)

    return jsonify({"generated_answer": output})

if __name__ == "__main__":
    app.run(debug=True)
