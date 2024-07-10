import streamlit as st
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain_community.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
with st.sidebar:
    st.title(' PDF Chat App')
    st.markdown('''
    ## About
    This app is used to upload our PDF and we can make query based on it and it will reply with the answers
    
    ''')
    add_vertical_space(5)
    st.write('Made by somanathan')

# Load environment variables if needed
load_dotenv()

# Function to handle main functionality
def main():
    st.header("Chat with PDF chatbot")
    
    # File uploader for PDF
    pdf = st.file_uploader("Upload Your PDF", type="pdf") 
    
    if pdf is not None:
       
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
       
        
              
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        
        query = st.text_input("Ask questions about your PDF file:")
        
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                
            st.write(response)

if __name__ == "__main__":
    main()
