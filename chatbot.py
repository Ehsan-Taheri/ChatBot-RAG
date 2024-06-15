import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def prepare_rag_llm(api_key, vector_store_name, temperature, max_length):
    instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vector_store_path = os.path.join("vector store", vector_store_name)
    loaded_db = FAISS.load_local(vector_store_path, instructor_embeddings, allow_dangerous_deserialization=True)
    
    llm = HuggingFaceHub(
        repo_id='tiiuae/falcon-7b-instruct',
        model_kwargs={"temperature": temperature, "max_length": max_length},
        huggingfacehub_api_token=api_key
    )
    
    memory = ConversationBufferWindowMemory(
        k=2, memory_key="chat_history", output_key="answer", return_messages=True
    )

    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        memory=memory
    )

    return qa_conversation

def generate_answer(question, token):
    if token == "":
        return "Insert the Hugging Face token", ["no source"]

    response = st.session_state.conversation({"question": question})
    answer = response.get("answer", "An error has occurred").split("Helpful Answer:")[-1].strip()
    explanation = response.get("source_documents", [])
    doc_source = [d.page_content for d in explanation]

    return answer, doc_source
