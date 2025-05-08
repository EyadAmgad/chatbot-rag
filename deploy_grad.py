# Imports
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace , HuggingFaceEndpoint
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Markdown, display
from data import documents
import os
# Access Tokens
hf_key = "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"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_key
# data

docs = [Document(page_content = text) for text in documents]
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
faiss_db = FAISS.from_documents(docs,embedding_model)
def retrieve_docs(query,k = 5):
    docs_faiss = faiss_db.similarity_search(query,k)
    return docs_faiss
query = "Why is staying hydrated important for both physical and mental performance?"
docs_faiss = retrieve_docs(query)
docs_faiss
llm = HuggingFaceEndpoint(repo_id = "microsoft/Phi-3.5-mini-instruct", task = "text_generation")
chat_model = ChatHuggingFace(llm = llm)
def ask(query,context):
    messages = [
        SystemMessage(content = "You are a scientist"),
        HumanMessage(content = f"""
                     Answer the {query} based on {context}
                     """ )
    ]
    response = chat_model.invoke(messages)
    return response.content
def ask_rag(query):
    context = retrieve_docs(query)
    return ask(query , context)
query = "Who are Pharaohs?"
response = ask_rag(query)
print(response)