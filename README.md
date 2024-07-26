# Contract-Advisor-RAG
This project is on build, evaluate and improve a RAG system for Contract Q&A (chatting with a contract and asking questions about the contract)

## key functionalities
Build simple Q&A pipeline with RAG using Langchain
Improve RAG systems by using different approaches
Build a RAG evaluation pipeline with RAGAS

## Features

RAG Framework: Leverages the LangChain framework to seamlessly integrate LLMs with contract data.

OpenAI GPT Models: Utilizes OpenAI's GPT models for natural language understanding and generation.

Web Interface: Provides a user-friendly web interface for users to interact with the system and ask questions about contract content.

FastAPI Backend: Uses FastAPI to create a high-performance backend server for handling user requests and allow interaction with the RAG pipeline.

Vector Database: Employs a vector database (e.g., Chroma) to store and retrieve contract information efficiently.

Re-ranking: Employs re-ranking mechanisms to prioritize the most contextually relevant information.

Evaluation using RAGAS: Includes a robust evaluation pipeline using metrics like accuracy, relevance, and response time to assess and improve performance.

## Project Structure
api/ : the backend connection

notebooks/ : includes different implementations and visualizations

utils/ : reusable python scripts

frontend/ : the frontend implementation using react

screenshots/ : screenshots of the project

test/: Includes unit and integration tests for the project.

## Installation

    git clone https://github.com/mahbubah/Contract-Advisor-RAG.git

In your terminal:

    cd Contract-Advisor-RAG
    pip install -r requirements.txt

Configure the environment

Add your OPENAI_API_KEY to a .env file in the project root directory.

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.