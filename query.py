import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DB_PATH = "../vectorstore"

def load_db():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)
    return db

def ask_question(db, query):
    docs = db.similarity_search(query, k=3)
    
    print("\n📄 Relevant context:\n")
    for doc in docs:
        print(doc.page_content[:300], "\n---\n")
    
    print("💡 Answer (based on documents):")
    print(" ".join([doc.page_content for doc in docs])[:500])

if __name__ == "__main__":
    db = load_db()
    
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        ask_question(db, query)