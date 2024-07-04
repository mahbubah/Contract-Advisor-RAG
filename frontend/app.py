import streamlit as st
import requests

# Function to upload a document via Flask API
def upload_document(file):
    files = {'file': file}
    response = requests.post('http://localhost:5000/api/upload', files=files)
    return response

# Function to query documents via Flask API
def query_documents(query, filename):
    data = {'query': query, 'filename': filename}
    response = requests.post('http://localhost:5000/api/query', json=data)
    return response

# Function to format RAG answer for chatbot-like display
def format_answer_for_chat(answer):
    return f"Bot: {answer}"

# Streamlit UI
def main():
    st.title('Contract Advisor Chatbot (RAG)')
    
    st.sidebar.title('Upload Document')
    uploaded_file = st.sidebar.file_uploader('Choose a .docx file', type=['docx'])

    if uploaded_file is not None:
        if st.sidebar.button('Upload'):
            with st.spinner('Uploading and processing document...'):
                response = upload_document(uploaded_file)

            if response.status_code == 200:
                data = response.json()
                st.sidebar.success(data['message'])
                st.sidebar.json(data)

                # Query section after successful upload
                st.header('Chat with Contract Advisor')
                query = st.text_input('You:')

                if st.button('Send'):
                    if query:
                        with st.spinner('Querying documents...'):
                            query_response = query_documents(query, data['filename'])

                        if query_response.status_code == 200:
                            answer = query_response.json().get('answer')
                            if answer:
                                st.text(format_answer_for_chat(answer))
                            else:
                                st.warning('No answer found.')
                        else:
                            st.error(f'Error {query_response.status_code}: {query_response.json()["error"]}')
                    else:
                        st.warning('Please enter a query.')

if __name__ == '__main__':
    main()
