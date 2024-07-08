from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from chromadb import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder
import numpy as np
from werkzeug.utils import secure_filename

# Load environment variables
_ = load_dotenv(find_dotenv())


# Set OpenAI API key
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

# Initialize ChromaDB client
chroma_client = Client()

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess documents
def preprocess_document(docx_path):
    doc = Document(docx_path)
    docx_texts = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]

    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], 
                                                        chunk_size=500, chunk_overlap=0)
    character_split_texts = character_splitter.split_text('\n\n'.join(docx_texts))

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts

# Endpoint for uploading a document
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the uploaded document
            documents = preprocess_document(filepath)

            # Embedding function
            embedding_function = SentenceTransformerEmbeddingFunction()

           
            chroma_collection = chroma_client.create_collection("Contract", embedding_function=embedding_function)
            chroma_collection = chroma_client.get_collection("Contract", embedding_function=embedding_function)

            # Add documents to the collection
            ids = [str(i) for i in range(len(documents))]
            chroma_collection.add(ids=ids, documents=documents)

            return jsonify({'message': 'Document uploaded and processed successfully'}), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400

# Function to query documents
def query_documents(query, collection):
    results = collection.query(query_texts=[query], n_results=5)
    retrieved_documents = results['documents'][0]
    return retrieved_documents

# Function to augment query with multiple questions
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

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content

# Function to run RAG pipeline
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
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

# Endpoint for querying using RAG pipeline
@app.route('/query', methods=['POST'])
def query_endpoint():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': "Query parameter 'query' is missing or empty"}), 400

    try:
        # Retrieve or create ChromaDB collection "Contract"
        embedding_function = SentenceTransformerEmbeddingFunction()
        chroma_collection = chroma_client.get_collection("Contract", embedding_function=embedding_function)

        # Execute RAG pipeline with user query
        response = rag_with_query_expansion(query, chroma_collection)
        return jsonify({'response': response}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main script
if __name__ == "__main__":
    app.run(debug=True)
