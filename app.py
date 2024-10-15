import streamlit as st
from PyPDF2 import PdfReader
import docx
import os
import warnings
import requests
import json
import base64
from langchain.schema import HumanMessage, AIMessage
from streamlit_chat import message
import logging  # Import logging module
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=r'.*Field "model_name".*')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

# Load environment variables
load_dotenv()

# Define the API key (Replace with your actual API key)
api_key = os.getenv("GENERATIVE_API_KEY")
if not api_key:
    st.error("Please set the GENERATIVE_API_KEY in your environment.")
    st.stop()

def main():
    configure_app()
    initialize_session_state()

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=False)
        process_button = st.button("Process")

    if process_button:
        if uploaded_files:
            process_files(uploaded_files)

    if st.session_state.processComplete:
        user_question = st.chat_input("Ask a question about your files.")
        if user_question:
            handle_user_input(user_question)
            # Process the API response immediately without spinner
            handle_api_response()

    display_chat_history()

def configure_app():
    st.set_page_config(page_title="Ask your PDF", layout="wide")
    st.header("Ask Your PDF")

def initialize_session_state():
    session_defaults = {
        "conversation": None,
        "chat_history": [],
        "processComplete": False,
        "response_complete": True,
        "cache_name": None,
        "files_text": "",
        "message_counter": 0,  # Add message counter
        "thinking_message_index": None,
        "pending_user_question": None,
    }
    for key, default in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def process_files(uploaded_file):
    files_text = extract_text_from_files(uploaded_file)
    st.session_state.files_text = files_text
    if files_text:
        st.session_state.processComplete = True
        if st.session_state.cache_name:
            update_cache_ttl()
        else:
            cache_uploaded_content(files_text)

def extract_text_from_files(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1]
    if ext == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    elif ext == ".docx":
        return extract_text_from_docx(uploaded_file)
    return ""

def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    return "".join(page.extract_text() or "" for page in pdf_reader.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return ' '.join(paragraph.text for paragraph in doc.paragraphs)

def cache_uploaded_content(files_text):
    encoded_text = base64.b64encode(files_text.encode('utf-8')).decode('utf-8')
    cache_api_url = 'https://generativelanguage.googleapis.com/v1beta/cachedContents?key=' + api_key
    cache_request_data = {
        "model": "models/gemini-1.5-flash-001",
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "text/plain",
                            "data": encoded_text
                        }
                    }
                ],
                "role": "user"
            }
        ],
        "systemInstruction": {
            "parts": [
                {
                    "text": "You are an expert in analyzing technical manuals."
                }
            ]
        },
        "ttl": "1200s"  # The time your cached document is saved is 20 minutes
    }
    headers = {'Content-Type': 'application/json'}
    logging.debug('API Request to cache content: %s %s', headers, cache_request_data)
    try:
        cache_response = requests.post(cache_api_url, headers=headers, data=json.dumps(cache_request_data))
        cache_response.raise_for_status()
        cache_response_data = cache_response.json()
        st.session_state.cache_name = cache_response_data.get('name')
        logging.debug('Cache response: %s', cache_response_data)
    except requests.exceptions.RequestException as e:
        st.error(f"Error caching content: {e}")
        logging.error('Error caching content: %s', e)
        st.stop()

def handle_user_input(user_question):
    # Assign a unique key to the message
    message_key = f"user_{st.session_state.message_counter}"
    st.session_state.chat_history.append({'type': 'human', 'content': user_question, 'key': message_key})
    st.session_state.message_counter += 1

    # Append 'Thinking...' as AI message
    ai_message_key = f"ai_{st.session_state.message_counter}"
    st.session_state.chat_history.append({'type': 'ai', 'content': 'Thinking...', 'key': ai_message_key})
    # Store the index of the 'Thinking...' message to update it later
    st.session_state.thinking_message_index = len(st.session_state.chat_history) - 1
    st.session_state.message_counter += 1

    st.session_state.response_complete = False
    st.session_state.pending_user_question = user_question

def handle_api_response():
    user_question = st.session_state.pending_user_question
    try:
        if st.session_state.cache_name:
            generate_api_url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-001:generateContent?key=' + api_key
            generate_request_data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": user_question
                            }
                        ],
                        "role": "user"
                    }
                ],
                "cachedContent": st.session_state.cache_name
            }
            headers = {'Content-Type': 'application/json'}
            logging.debug('API Request to generate content: %s %s', headers, generate_request_data)
            generate_response = requests.post(generate_api_url, headers=headers, data=json.dumps(generate_request_data))
            generate_response.raise_for_status()
            generate_response_data = generate_response.json()
            logging.debug('API Response: %s', generate_response_data)
            # Extract AI response from API response
            ai_response = generate_response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            # Update the 'Thinking...' message with AI response
            st.session_state.chat_history[st.session_state.thinking_message_index]['content'] = ai_response
        else:
            st.error("No cached content available.")
            ai_response = "No cached content available."
            # Update the 'Thinking...' message
            st.session_state.chat_history[st.session_state.thinking_message_index]['content'] = ai_response
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating content: {e}")
        logging.error('Error generating content: %s', e)
        # Update the 'Thinking...' message with error message
        st.session_state.chat_history[st.session_state.thinking_message_index]['content'] = f"Error generating content: {e}"

    # Set response complete flag and clear pending user question
    st.session_state.response_complete = True
    st.session_state.pending_user_question = None

def update_cache_ttl():
    patch_api_url = f"https://generativelanguage.googleapis.com/v1beta/{st.session_state.cache_name}?key=" + api_key
    patch_request_data = {
        "ttl": "1200s"
    }
    headers = {'Content-Type': 'application/json'}
    logging.debug('API Request to update TTL: %s %s', headers, patch_request_data)
    try:
        patch_response = requests.patch(patch_api_url, headers=headers, data=json.dumps(patch_request_data))
        patch_response.raise_for_status()
        logging.debug('TTL Update Response: %s', patch_response.text)
    except requests.exceptions.RequestException as e:
        st.error(f"Error updating cache TTL: {e}")
        logging.error('Error updating cache TTL: %s', e)

def display_chat_history():
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for message_data in st.session_state.chat_history:
            if message_data['type'] == 'human':
                message(message_data['content'], is_user=True, key=message_data['key'])
            elif message_data['type'] == 'ai':
                message(message_data['content'], key=message_data['key'])

    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.markdown(f"<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
