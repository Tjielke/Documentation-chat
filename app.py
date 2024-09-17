from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Facebook AI similarity search
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import StdOutCallbackHandler
from streamlit_chat import message
import docx
import os
import warnings
warnings.filterwarnings("ignore", message=r'.*Field "model_name".*')


# Main function to load the application
def main():
    load_dotenv()
    configure_app()

    # Initialize session state variables if not already present
    initialize_session_state()

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        process_button = st.button("Process")

    # Process the files if the user clicks the process button
    if process_button:
        process_files(uploaded_files)

    # Handle user questions after processing the files
    if st.session_state.processComplete:
        user_question = st.chat_input("Ask a question about your files.")
        if user_question:
            handle_user_input(user_question)

# Configure Streamlit application
def configure_app():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask Your PDF")

# Initialize session state variables
def initialize_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

# Process uploaded files, split text into chunks, and initialize conversation chain
def process_files(uploaded_files):
    files_text = extract_text_from_files(uploaded_files)
    text_chunks = split_text_into_chunks(files_text)
    vectorstore = create_vectorstore(text_chunks)
    st.session_state.conversation = create_conversation_chain(vectorstore)
    st.session_state.processComplete = True

# Extract text from multiple files
def extract_text_from_files(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1]
        if ext == ".pdf":
            text += extract_text_from_pdf(uploaded_file)
        elif ext == ".docx":
            text += extract_text_from_docx(uploaded_file)
    return text

# Extract text from a PDF file
def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    return "".join(page.extract_text() for page in pdf_reader.pages)

# Extract text from a DOCX file
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return ' '.join(paragraph.text for paragraph in doc.paragraphs)

# Split large text into smaller chunks for better processing
def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

# Create a FAISS vectorstore using HuggingFace embeddings
def create_vectorstore(text_chunks):
    # Specify a model name explicitly
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embeddings)

# Create the conversation chain for handling user queries
def create_conversation_chain(vectorstore):
    handler = StdOutCallbackHandler()
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 5, "max_length": 64})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        callbacks=[handler]
    )

# Handle user input and respond with the conversation chain
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    display_chat_history()

# Display the chat history in the Streamlit chat interface
def display_chat_history():
    response_container = st.container()
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))

# Run the main application
if __name__ == '__main__':
    main()
