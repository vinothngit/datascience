#1. Read the documents from different folder
#2. Create embedding for each folder with unique id for easy retrievals
#3.appending the embeddings

#import getpass
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
langchain_api_key=os.environ['LANGCHAIN_API_KEY']
#print(langchain_api_key)

groq_api_key= os.environ['GROQ_API_KEY']

#from langchain_groq import ChatGroq

#llm = ChatGroq(model="llama3-8b-8192")


#Function to read documents
def load_docs(directory):
  loader = PyPDFDirectoryLoader(directory)
  documents = loader.load()
  return documents


# Passing the directory to the 'load_docs' function
#directory = 'data/'
#documents = load_docs(directory)

#print(len(documents))

#split the documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

#docs = split_docs(documents)
print("\n \n Successfully Completed Data Preparation")
