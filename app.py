from flask import Flask
from Api.logging import configure_logging
from Api.endpoints import api_bp
from Api.utils import create_chromadb_collection
from chromadb import Client

app = Flask(__name__)
configure_logging()

# Initialize ChromaDB client (if necessary)
chroma_client = Client()

# Do not initialize chroma_collection here

app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)
