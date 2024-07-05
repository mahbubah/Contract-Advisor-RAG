import streamlit as st
import requests
import os

# Set up API endpoint URLs
UPLOAD_URL = "http://localhost:5000/upload"
QUERY_URL = "http://localhost:5000/query"

# Function to upload file to Flask API
def upload_file(file):
    files = {'file': file}
    try:
        response = requests.post(UPLOAD_URL, files=files)
        if response.status_code == 200:
            st.success("Document uploaded and processed successfully!")
        else:
            st.error(f"Upload failed with error: {response.json()['error']}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error during upload: {e}")

# Function to query document using Flask API
def query_document(query):
    payload = {'query': query}
    try:
        response = requests.post(QUERY_URL, json=payload)
        if response.status_code == 200:
            return response.json()['response']
        else:
            st.error(f"Query failed with error: {response.json()['error']}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error during query: {e}")
        return None

# Main Streamlit application
def main():
    st.title("Contract Document Chatbot")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # File upload section
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a document (.docx)", type="docx")

    if st.button("Upload") and uploaded_file:
        upload_file(uploaded_file)

    # Query section
    st.header("Ask a Question")
    query = st.text_area("Enter your question")

    if st.button("Ask") and query:
        response = query_document(query)
        if response:
            st.session_state.chat_history.append((query, response))

    # Display chat history
    st.header("Chat History")
    for idx, (q, a) in enumerate(st.session_state.chat_history):
        st.text(f"Question {idx + 1}: {q}")
        st.text(f"Answer {idx + 1}: {a}")
        st.markdown("---")

if __name__ == "__main__":
    main()
