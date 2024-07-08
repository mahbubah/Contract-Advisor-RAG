import streamlit as st
import requests

# Define API endpoint URLs
UPLOAD_URL = 'http://localhost:5000/upload'  # Replace with your Flask server address
QUERY_URL = 'http://localhost:5000/query'    # Replace with your Flask server address

# Custom CSS styles
custom_styles = """
<style>
body {
    background-color: #f0f0f0;  /* Light gray background */
    color: #333333;  /* Dark text color */
}
.button-primary {
    background-color: #0069d9;  /* Blue primary color */
    color: white;  /* Text color */
    border-color: #0069d9;  /* Border color */
    padding: 0.5rem 1rem;  /* Padding */
    font-size: 1rem;  /* Font size */
    cursor: pointer;  /* Cursor style */
}
</style>
"""

# Streamlit App
def main():
    st.title('Contract Advisor Chatbot')

    # Inject custom styles into the page
    st.markdown(custom_styles, unsafe_allow_html=True)

    st.image('https://cdn.pixabay.com/photo/2018/04/10/17/18/conversation-3300430_960_720.png', use_column_width=True)

    st.markdown('Welcome to the Contract Advisor Chatbot! Upload a document or ask a question below.')

    st.header('Upload a Document')
    uploaded_file = st.file_uploader('Upload a DOCX file', type=['docx'])

    if uploaded_file is not None:
        st.write('File uploaded successfully!')

        if st.button('Process Document', key='process_button'):
            files = {'file': uploaded_file}
            response = requests.post(UPLOAD_URL, files=files)

            if response.status_code == 200:
                st.success('Document processed and uploaded successfully!')
            else:
                st.error(f'Error: {response.json()["error"]}')

    st.header('Ask a Question')
    query = st.text_area('Enter your question here')

    if st.button('Submit', key='submit_button'):
        if query:
            try:
                response = requests.post(QUERY_URL, json={'query': query})

                if response.status_code == 200:
                    answer = response.json()['response']
                    st.success('Here is the answer:')
                    st.write(answer)
                else:
                    st.error(f'Error: {response.json()["error"]}')

            except requests.exceptions.RequestException as e:
                st.error(f'Request failed: {e}')

if __name__ == '__main__':
    main()
