from flask import Blueprint, request, jsonify
from .utils import preprocess_document_and_save, fetch_from_postgres, rag_with_query_expansion,save_uploaded_file,create_chromadb_collection,chroma_client

api_bp = Blueprint('api', __name__)

@api_bp.route('/upload', methods=['POST'])
def upload_endpoint():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        # Save the uploaded file and get its path
        docx_path = save_uploaded_file()
        
        # Preprocess the uploaded document and save to PostgreSQL
        documents = preprocess_document_and_save(docx_path)
        
        if not documents:
            return jsonify({'error': 'No valid documents found in the uploaded file'}), 400
        
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
        filename = request_data.get('filename')  # Get filename from request
        
        if not query:
            return jsonify({'error': 'Query parameter is missing or empty'}), 400
        
        if not filename:
            return jsonify({'error': 'Filename parameter is missing or empty'}), 400

        # Fetch documents from PostgreSQL
        documents = fetch_from_postgres(filename)

        if not documents:
            return jsonify({'error': f'No documents found for filename: {filename}'}), 404

        collection_name = 'contract'
        #if not documents:
        #documents = preprocess_document_and_save(filename)
        create_chromadb_collection(collection_name, documents)

        # Query ChromaDB for relevant documents
        collection = chroma_client.get_collection(collection_name)
        results = rag_with_query_expansion(query, collection=collection)

        return jsonify({'answer': results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
