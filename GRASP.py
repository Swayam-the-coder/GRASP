import streamlit as st
from streamlit_option_menu import option_menu
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import bs4
import speech_recognition as sr
from sqlalchemy import create_engine
import pandas as pd
import requests

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Set page config
st.set_page_config(page_title='🤖 GRASP', layout='wide', initial_sidebar_state='expanded')

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #E8F6F3;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #45A049;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            border: 1px solid #ccc;
            padding: 10px;
        }
        .stSidebar>div>div>div>div {
            background-color: #E8F6F3;
        }
        .stFileUploader>label>div>div>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
        }
        .stFileUploader>label>div>div>button:hover {
            background-color: #45A049;
        }
    </style>
""", unsafe_allow_html=True)

def home_page():
    st.markdown("<h1 style='text-align: center;'> 🤖 Generative Retrieval Augmented Search Platform</h1>",unsafe_allow_html=True)
    st.header("Welcome to GRASP")
    st.subheader("Explore and learn about each RAG method on their respective pages.")

def pdf_rag_page():
    st.title('PDF RAG 📄')
    st.sidebar.header("Upload your PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

    st.write("### Instructions 📜")
    st.write("""
        1. Upload a PDF file using the sidebar.
        2. Enter your query in the text input below.
        3. Click the 'Get Results' button to process the PDF and provide relevant answers based on the content.
    """)

    def load_and_process_pdf(file):
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(file.name)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    rag_chain = None
    if uploaded_file:
        rag_chain = load_and_process_pdf(uploaded_file)

    input_text = st.text_input("Please feel free to ask any doubts! 📝")

    if st.button("Get Results"):
        if input_text and rag_chain:
            with st.spinner('Processing...'):
                try:
                    response = rag_chain.invoke({"input": input_text})
                    st.write(response["answer"])
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        elif input_text:
            st.warning("Please upload a PDF file to ask questions.")

def web_rag_page():
    st.title('Web RAG 🌐')
    st.sidebar.header("Enter a web URL")
    url = st.sidebar.text_input("URL")

    st.write("### Instructions 📜")
    st.write("""
        1. Enter a web URL using the sidebar.
        2. Enter your query in the text input below.
        3. Click the 'Get Results' button to process the webpage content and provide relevant answers based on the content.
    """)

    def load_and_process_web(url):
        loader = WebBaseLoader(web_paths=(url,), bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("mw-body-content", "mw-headline"))))
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    rag_chain = None
    if url:
        rag_chain = load_and_process_web(url)

    input_text = st.text_input("Please feel free to ask any doubts! 📝")

    if st.button("Get Results"):
        if input_text and rag_chain:
            with st.spinner('Processing...'):
                try:
                    response = rag_chain.invoke({"input": input_text})
                    st.write(response["answer"])
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        elif input_text:
            st.warning("Please enter a URL to ask questions.")

def text_document_rag_page():
    st.title('Text Document RAG 📄')
    st.sidebar.header("Upload your Text Document")
    uploaded_file = st.sidebar.file_uploader("Choose a text file", type=["txt"])

    st.write("### Instructions 📜")
    st.write("""
        1. Upload a text file using the sidebar.
        2. Enter your query in the text input below.
        3. Click the 'Get Results' button to process the text content and provide relevant answers based on the content.
    """)

    def load_and_process_text(file):
        content = file.read().decode('utf-8')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(content)
        vectorstore = Chroma.from_texts(texts=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    rag_chain = None
    if uploaded_file:
        rag_chain = load_and_process_text(uploaded_file)

    input_text = st.text_input("Please feel free to ask any doubts! 📝")

    if st.button("Get Results"):
        if input_text and rag_chain:
            with st.spinner('Processing...'):
                try:
                    response = rag_chain.invoke({"input": input_text})
                    st.write(response["answer"])
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        elif input_text:
            st.warning("Please upload a text file to ask questions.")

def audio_rag_page():
    st.title('Audio RAG 🎤')
    st.sidebar.header("Upload your Audio")
    uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav"])

    st.write("### Instructions 📜")
    st.write("""
        1. Ensure your audio file is in a supported format (PCM WAV, AIFF/AIFF-C, or Native FLAC).
        2. Upload an audio file using the sidebar.
        3. Enter your query in the text input below.
        4. Click the 'Get Results' button to process the text extracted from the audio and provide relevant answers based on the content.
    """)

    def load_and_process_audio(file):
        recognizer = sr.Recognizer()
        try:
            audio_file = sr.AudioFile(file)
        except ValueError:
            st.error("Audio file could not be read as PCM WAV, AIFF/AIFF-C, or Native FLAC; check if file is corrupted or in another format.")
            return None
        with audio_file as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(text)
        vectorstore = Chroma.from_texts(texts=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    rag_chain = None
    if uploaded_file:
        rag_chain = load_and_process_audio(uploaded_file)

    input_text = st.text_input("Please feel free to ask any doubts! 📝")

    if st.button("Get Results"):
        if input_text and rag_chain:
            with st.spinner('Processing...'):
                try:
                    response = rag_chain.invoke({"input": input_text})
                    st.write(response["answer"])
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        elif input_text:
            st.warning("Please upload an audio file to ask questions.")

def database_rag_page():
    st.title('Database RAG 🗄️')
    st.sidebar.header("Enter Database Credentials")
    db_url = st.sidebar.text_input("Database URL")
    table_name = st.sidebar.text_input("Table Name")

    st.write("### Instructions 📜")
    st.write("""
        1. Enter the database URL and table name using the sidebar.
        2. Enter your query in the text input below.
        3. Click the 'Get Results' button to process the data from the specified table and provide relevant answers based on the content.
    """)

    def load_and_process_db(db_url, table_name):
        engine = create_engine(db_url)
        df = pd.read_sql_table(table_name, engine)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(df.to_string())
        vectorstore = Chroma.from_texts(texts=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    rag_chain = None
    if db_url and table_name:
        rag_chain = load_and_process_db(db_url, table_name)

    input_text = st.text_input("Please feel free to ask any doubts! 📝")

    if st.button("Get Results"):
        if input_text and rag_chain:
            with st.spinner('Processing...'):
                try:
                    response = rag_chain.invoke({"input": input_text})
                    st.write(response["answer"])
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        elif input_text:
            st.warning("Please enter database credentials to ask questions.")

def api_rag_page():
    st.title('API RAG 🔌')
    st.sidebar.header("Enter API Endpoint")
    api_url = st.sidebar.text_input("API URL")

    st.write("### Instructions 📜")
    st.write("""
        1. Enter the API URL using the sidebar.
        2. Enter your query in the text input below.
        3. Click the 'Get Results' button to process the data from the specified API endpoint and provide relevant answers based on the content.
    """)

    def load_and_process_api(api_url):
        response = requests.get(api_url)
        data = response.json()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(str(data))
        vectorstore = Chroma.from_texts(texts=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    rag_chain = None
    if api_url:
        rag_chain = load_and_process_api(api_url)

    input_text = st.text_input("Please feel free to ask any doubts! 📝")

    if st.button("Get Results"):
        if input_text and rag_chain:
            with st.spinner('Processing...'):
                try:
                    response = rag_chain.invoke({"input": input_text})
                    st.write(response["answer"])
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        elif input_text:
            st.warning("Please enter an API URL to ask questions.")

# Extend the navigation menu
with st.sidebar:
    selected = option_menu(
        "🤖 GRASP",
        ["Home", "PDF RAG 📄", "Web RAG 🌐", "Text Document RAG 📄", "Audio RAG 🎤", "Database RAG 🗄️", "API RAG 🔌"],
        icons=["house", "file-earmark-pdf", "globe", "file-earmark-text", "mic", "database", "plug"],
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )

# Display the selected page
if selected == "Home":
    home_page()
elif selected == "PDF RAG 📄":
    pdf_rag_page()
elif selected == "Web RAG 🌐":
    web_rag_page()
elif selected == "Text Document RAG 📄":
    text_document_rag_page()
elif selected == "Audio RAG 🎤":
    audio_rag_page()
elif selected == "Database RAG 🗄️":
    database_rag_page()
elif selected == "API RAG 🔌":
    api_rag_page()

# Additional User Feedback Section
st.sidebar.header("User Feedback")
feedback = st.sidebar.text_area("Provide your feedback here:")
if st.sidebar.button("Submit Feedback"):
    with open("feedback.txt", "a") as f:
        f.write(f"Feedback: {feedback}\n")
    st.sidebar.success("Feedback submitted successfully!")
