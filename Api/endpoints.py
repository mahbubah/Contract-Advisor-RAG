from flask import Blueprint, request, jsonify, current_app
from chromadb import Client
from .utils import preprocess_document, rag_with_query_expansion, save_uploaded_file, create_chromadb_collection

api_bp = Blueprint('api', __name__)
chroma_client = Client()

@api_bp.route('/upload', methods=['POST'])
def upload_endpoint():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        # Save the uploaded file and get its path
        docx_path = save_uploaded_file()
        
        # Preprocess the uploaded document
        documents = preprocess_document(docx_path)
        
        if not documents:
            return jsonify({'error': 'No valid documents found in the uploaded file'}), 400
        
        # Create or update the ChromaDB collection with the new documents
        chroma_collection = create_chromadb_collection("contract", documents)
        
        return jsonify({'message': 'Document uploaded and processed successfully', 'documents': documents}), 200
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400  # Handle specific errors like empty documents
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Handle generic errors

@api_bp.route('/query', methods=['POST'])
def query_endpoint():
    try:
        request_data = request.get_json()
        query = request_data.get('query')  # Use .get() to safely retrieve 'query'
        if not query:
            return jsonify({'error': 'Query parameter is missing or empty'}), 400

        results = rag_with_query_expansion(query, collection=chroma_collection)
        return jsonify({'answer': results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
