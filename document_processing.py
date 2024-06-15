import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import shutil

def read_pdf(file):
    document = ""
    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()
    return document

def read_txt(file):
    document = str(file.read(), 'utf-8')
    return document

def split_doc(document, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split = splitter.split_text(document)
    split = splitter.create_documents(split)
    return split


def embedding_storing(split, create_new_vs, existing_vector_store, new_vs_name):
    instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(split, instructor_embeddings)

    if create_new_vs:
        new_vs_path = os.path.join("vector store", new_vs_name)
        os.makedirs(new_vs_path)
        db.save_local(new_vs_path)
    else:
        load_db = FAISS.load_local(os.path.join("vector store", existing_vector_store), instructor_embeddings, allow_dangerous_deserialization=True)
        load_db.merge_from(db)
        load_db.save_local(os.path.join("vector store", existing_vector_store))

    st.success("The document has been saved.")

