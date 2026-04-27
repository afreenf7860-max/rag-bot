import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DATA_PATH = "../data"
DB_PATH = "../vectorstore"


def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_PATH
    )

    db.persist()


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} pages")

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    create_vectorstore(chunks)

    print("Indexing completed!")