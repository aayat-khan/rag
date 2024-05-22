import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()

# Set up Streamlit
st.title("Interactive Book Search")
st.write("Upload a PDF and ask questions about its content:")

# Function to load and process the document
def load_and_process_document(pdf_file_path):
    pdfloader = PyPDFLoader(pdf_file_path)
    mydocuments = pdfloader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(mydocuments)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Process the uploaded PDF file
    db = load_and_process_document(tmp_file_path)
    retriever = db.as_retriever()
    tool = create_retriever_tool(
        retriever,
        "pdf_search",  # Simple and valid function name
        "Searches and returns documents regarding the uploaded PDF"
    )
    tools = [tool]

    # Set up the conversational agent
    llm = ChatOpenAI(temperature=0)
    agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

    # Input field for user query
    user_query = st.text_input("Enter your question about the PDF content:")

    # Button to submit query
    if st.button("Submit"):
        if user_query:
            input = {"input": user_query}
            try:
                result = agent_executor.invoke(input)
                st.write("### Answer:")
                st.write(result["output"])
            except Exception as e:
                st.write("### Error:")
                st.write(str(e))
        else:
            st.write("Please enter a question.")
else:
    st.write("Please upload a PDF file to proceed.")
