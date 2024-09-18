from langchain_community.vectorstores import Pinecone 
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pinecone import Pinecone as PineconeClient #Importing the Pinecone class from the pinecone package
from langchain_community.vectorstores import Pinecone
import os
from data_ingestion import load_docs, split_docs

from langchain_huggingface import HuggingFaceEmbeddings

HUGGINGFACEHUB_API_TOKEN=os.environ["HUGGINGFACEHUB_API_TOKEN"]
PINECONE_API_KEY=os.getenv("‘PINECONE_API_KEY’")

# Set your Pinecone API key
# Recent changes by langchain team, expects ""PINECONE_API_KEY" environment variable for Pinecone usage! So we are creating it here
# we are setting the environment variable "PINECONE_API_KEY" to the value and in the next step retrieving it :)

def vector_db():
    
    directory = 'data/'
    documents = load_docs(directory)
    docs=split_docs(documents, chunk_size=1000, chunk_overlap=20)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize the Pinecone client
    PineconeClient(api_key=PINECONE_API_KEY, environment="us-east-1-aws-starter")
    index_name="rag"
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    print("Vectorization of document completed and loaded")
    return index


if __name__=="__main__":

    vector_db()
