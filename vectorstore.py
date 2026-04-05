from embebddings import get_embeddings
from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS

from typing import List
from langchain_community.document_loaders import CSVLoader


def load_document()-> List[Document]:
    

    csv_loader = CSVLoader(
        file_path="./libs/dataset.csv",
        encoding="utf-8"
    )

    csv_docs = csv_loader.load()

    return [
        Document( 
            page_content=doc.page_content,
            metadata={
                'id': doc.metadata.get('row')+1,
            }
        )
        for doc in csv_docs
    ]

def embedding(docs: List[Document]):
    embeddings= get_embeddings()
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    return vectorstore

def save_vector_to_local(vectorstore):
    path_str = './exp-faiss'
    vectorstore.save_local(path_str)


def load_vector_from_local():
    path_str = './exp-faiss'
    return FAISS.load_local(
        path_str,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

def init_vectorstore():
    docs = load_document()
    vectorstore = embedding(docs)
    save_vector_to_local(vectorstore)
    return vectorstore