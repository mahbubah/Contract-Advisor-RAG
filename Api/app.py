from flask import Flask, request, jsonify
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from chromadb import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import openai
from sentence_transformers import CrossEncoder
import numpy as np
import os

app = Flask(__name__)

# Load environment variables
openai.api_key = os.environ['OPENAI_API_KEY']
chroma_client = Client()

# Define global variables
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'docx'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Preprocess the uploaded document
            documents = preprocess_document(filepath)

            # Create ChromaDB collection and add documents
            collection_name = 'contract'
            create_chromadb_collection(collection_name, documents)

            return jsonify({'message': 'Document uploaded and processed successfully', 'documents': documents}), 200
        else:
            return jsonify({'error': 'Invalid file extension or file format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_document():
    try:
        request_data = request.get_json()
        query = request_data.get('query')
        if not query:
            return jsonify({'error': 'Query parameter is missing or empty'}), 400

        # Fetch documents from ChromaDB
        collection_name = 'contract'
        collection = chroma_client.get_collection(collection_name)
        results = rag_with_query_expansion(query, collection=collection)

        return jsonify({'answer': results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

def create_chromadb_collection(collection_name, documents):
    if not documents:
        raise ValueError("No documents provided to create collection")

    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_collection = chroma_client.create_collection(collection_name, embedding_function=embedding_function)
    ids = [str(i) for i in range(len(documents))]
    chroma_collection.add(ids=ids, documents=documents)

def rag_with_query_expansion(original_query, collection_name='contract', model="gpt-3.5-turbo"):
    # Expand the original query
    expanded_queries = augment_multiple_query(original_query, model=model)[:5]  # Limit to top 5 expanded queries

    # Include the original query in the list of queries
    queries = [original_query] + expanded_queries

    # Query ChromaDB for relevant documents
    collection = chroma_client.get_collection(collection_name)
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

if __name__ == '__main__':
    app.run(debug=True)
