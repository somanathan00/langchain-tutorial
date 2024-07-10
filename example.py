from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from constants import openai_key
import os

# Load environment variables
os.environ["OPENAI_API_KEY"] = openai_key

# Read PDF and extract text
PdfReader = PdfReader('./gokhul_resume.pdf')
raw_text = ''
for i, page in enumerate(PdfReader.pages):
    content = page.extract_text()
    print(content)
    if content:
        raw_text += content

print(raw_text)

# Split text into manageable chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)
print(f"Number of text chunks: {len(texts)}")

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Embed documents using OpenAIEmbeddings
embedded_texts = embeddings.embed_documents(texts)

# Create FAISS vector store from embedded texts
document_search = FAISS.from_texts (texts,embedding=embeddings)

# Load question answering chain
chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")

# Example query for question answering
query = "how much he scored in higher secondary"

# Perform similarity search and run question answering chain
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)
