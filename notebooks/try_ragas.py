import os
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from chromadb import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv, find_dotenv
import openai
from sentence_transformers import CrossEncoder
import numpy as np
from openai import OpenAI

# Load environment variables
_ = load_dotenv(find_dotenv())

# Set OpenAI API key
#openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

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

# Example document path
docx_path = "Robinson Advisory.docx"
documents = preprocess_document(docx_path)

# Embedding function
embedding_function = SentenceTransformerEmbeddingFunction()

# Create ChromaDB collection and add documents
chroma_collection = chroma_client.create_collection("Contract", embedding_function=embedding_function)
ids = [str(i) for i in range(len(documents))]
chroma_collection.add(ids=ids, documents=documents)

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

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content

def rag_with_query_expansion(original_query, model="gpt-3.5-turbo"):
    # Expand the original query
    expanded_queries = augment_multiple_query(original_query, model=model)[:5]  # Limit to top 5 expanded queries

    # Include the original query in the list of queries
    queries = [original_query] + expanded_queries

    # Query ChromaDB for relevant documents
    results = chroma_collection.query(query_texts=queries, n_results=5, include=['documents'])
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
    print(information)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert legal contract advisor assistant. Your users are asking questions about information contained in the contract."
                       "You will be shown the user's question, and the relevant information from the contract. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {original_query}. \n Information: {information}"},
        
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

def retrieve_documents(query):
    # Perform your retrieval logic here
    results = chroma_collection.query(query_texts=[query], n_results=5)
    retrieved_documents = results['documents'][0]  # Assuming the structure of returned documents
    return retrieved_documents

from datasets import Dataset

# Define your questions, answers, and ground truths
questions = [
    "Is there a non-compete obligation to the Advisor?",
    "Can the Advisor charge for meal time?",
    "In which street does the Advisor live?",
    "Is the Advisor entitled to social benefits?",
]

ground_truths = [
    "Yes. During the term of engagement with the Company and for a period of 12 months thereafter.",
    "No. See Section 6.1, Billable Hour doesn’t include meals or travel time.",
    "1 Rabin st, Tel Aviv, Israel",
    "No. According to section 8 of the Agreement, the Advisor is an independent consultant and shall not be entitled to any overtime pay, insurance, paid vacation, severance payments or similar fringe or employment benefits from the Company.",
]

answers = []
contexts = []


for query in questions:
    answers.append(rag_with_query_expansion(query))
    relevant_documents = retrieve_documents(query)
    #contexts.append([doc.page_content for doc in relevant_documents])
    contexts.append(relevant_documents)  # Already contains text content
    #contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# Create the dataset dictionary
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths  
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)


for i in range(3):  
    print(dataset[i])

import asyncio
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

async def run_evaluation():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)  # Set the loop for this context
    try:
        result = await evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            raise_exceptions=False
        )
    finally:
        loop.close()  # Close the loop after use

    df = result.to_pandas()
    print(df)

# Run the asynchronous function using asyncio.run
asyncio.run(run_evaluation())

