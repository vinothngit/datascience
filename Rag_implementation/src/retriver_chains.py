#from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from data_ingestion import load_docs, split_docs
from vectorstore_pinecone import vector_db
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
#llm = Ollama(model='llama3.1')

print(llm)

# create a prompt that can be passed to the llm , that prompt takes the context 

prompt = ChatPromptTemplate.from_template(""" 
Answer the following question based on the provided context.
Think step by step before providing a detailed answer.
                                          <context>
                                          {context}
                                          </context>
                                          Question: {input}
                                          
""")

# chains , we should use mainly when we come across prompts,
# as our objective is to retrive the context, then pass prompt to LLM and generate answers using the contect and prompt
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm, prompt=prompt)


# retrievers
"""
Retrievers: A retriever is an interface that returns documents given
 an unstructured query. It is more general than a vector store.
 A retriever does not need to be able to store documents, only to 
 return (or retrieve) them. Vector stores can be used as the backbone
 of a retriever, but there are other types of retrievers as well. 
 https://python.langchain.com/docs/modules/data_connection/retrievers/   
"""

db =vector_db()
retriever = db.as_retriever()
#print(retriever)

# Retrieval chains:
"""
Retrieval chain:This chain takes in a user inquiry, which is then
passed to the retriever to fetch relevant documents. Those documents 
(and original inputs) are then passed to an LLM to generate a response
https://python.langchain.com/docs/modules/chains/
"""

from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever,document_chain)
response = retrieval_chain.invoke({"input": "Explain LORA to me in 5 sentences"})
print(response['answer'])

