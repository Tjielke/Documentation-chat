from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StdOutCallbackHandler
from langchain.schema import HumanMessage, AIMessage
from streamlit_chat import message
import torch
import docx
import os
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

warnings.filterwarnings("ignore", message=r'.*Field "model_name".*')

# Caching the model loading process
@st.cache_resource
def load_model():
    token = "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"
    # tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF",token = token, device_map="cpu")
    # Load the LLaMA 2 tokenizer (compatible with Mistral)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True,device_map='balanced',torch_dtype=torch.float8_e4m3fn,token=token)
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True,device_map='balanced',torch_dtype=torch.float8_e4m3fn,token=token)
    return tokenizer, model

def main():
    
    configure_app()
    initialize_session_state()

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        process_button = st.button("Process")

    if process_button:
        process_files(uploaded_files)

    if st.session_state.processComplete:
        user_question = st.chat_input("Ask a question about your files.")
        if user_question:
            handle_user_input(user_question)

    # Always display chat history, as it's controlled by `response_complete` internally
    display_chat_history()

def configure_app():
    st.set_page_config(page_title="Ask your PDF", layout="wide")
    st.header("Ask Your PDF")

def initialize_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False
    if "response_complete" not in st.session_state:
        st.session_state.response_complete = True  # Initialize as True so the UI doesn't block at start

def process_files(uploaded_files):
    files_text = extract_text_from_files(uploaded_files)
    text_chunks = split_text_into_chunks(files_text)
    vectorstore = create_vectorstore(text_chunks)
    st.session_state.conversation = create_conversation_chain(vectorstore)
    st.session_state.processComplete = True

def extract_text_from_files(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1]
        if ext == ".pdf":
            text += extract_text_from_pdf(uploaded_file)
        elif ext == ".docx":
            text += extract_text_from_docx(uploaded_file)
    return text

def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    return "".join(page.extract_text() for page in pdf_reader.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return ' '.join(paragraph.text for paragraph in doc.paragraphs)

def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,  # Reduced chunk size
        chunk_overlap=50,  # Adjusted overlap
        length_function=len
    )
    return text_splitter.split_text(text)

def create_vectorstore(text_chunks):
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embeddings)

def create_conversation_chain(vectorstore):
    handler = StdOutCallbackHandler()
    
    # Use the cached model
    tokenizer, model = load_model()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_tokens=512,min_length=100)
    llm = HuggingFacePipeline(pipeline=pipe)
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        callbacks=[handler]
    )

def handle_user_input(user_question):
    # Immediately append the user's question to the chat history
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    
    # Set response_complete to False to indicate that the AI is processing the response
    st.session_state.response_complete = False
    # Display the chat history so far
    display_chat_history()

    # Get the AI's response
    response = st.session_state.conversation({'question': user_question})
    
    # Append the AI's response to the chat history
    st.session_state.chat_history.extend(response['chat_history'][-1:])  # Add only the latest AI message
    
    # Set response_complete to True to allow re-rendering of the updated chat
    st.session_state.response_complete = True

def display_chat_history():
    # Always display the chat history, but limit re-renders by checking `response_complete`
    if not st.session_state.response_complete:
        return

    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for i, messages in enumerate(st.session_state.chat_history):
            if isinstance(messages, HumanMessage):
                message(messages.content, is_user=True, key=f"user_{i}")
            elif isinstance(messages, AIMessage):
                message(messages.content, key=f"bot_{i}")
                
        # Add an invisible element to maintain consistent height
        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

    # Ensure scrolling to the bottom to keep up with new messages
    st.markdown(f"<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
