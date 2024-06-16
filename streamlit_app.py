import os
import streamlit as st
from document_processing import read_pdf, read_txt, split_doc, embedding_storing
from chatbot import prepare_rag_llm, generate_answer

def load_secrets():
    api_key = st.text_input("Enter your API Key:", type='password')
    return api_key

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Select a page:", ["Document Embedding", "RAG Chatbot"])

    if selection == "Document Embedding":
        display_document_embedding_page()
    elif selection == "RAG Chatbot":
        display_chatbot_page()

def display_document_embedding_page():
    vector_store_list = [item for item in os.listdir("vector store") if item != ".DS_Store"]

    st.title("Document Embedding")
    st.markdown("Upload documents to create a custom knowledge base for the chatbot.")

    with st.form("document_input"):
        document = st.file_uploader("Upload Documents (PDF or TXT)", type=['pdf', 'txt'], accept_multiple_files=True)
        instruct_embeddings = st.text_input("Instruct Embeddings Model", value="sentence-transformers/all-MiniLM-L6-v2")
        chunk_size = st.number_input("Chunk Size", value=200, min_value=0, step=1)
        chunk_overlap = st.number_input("Chunk Overlap", value=10, min_value=0, step=1)

        if not os.path.exists("vector store"):
            os.makedirs("vector store")
        vector_store_list = ["<New>"] + vector_store_list
        existing_vector_store = st.selectbox("Select or Create Vector Store", vector_store_list)
        new_vs_name = st.text_input("New Vector Store Name", value="new_vector_store")
        save_button = st.form_submit_button("Save Vector Store")

    if save_button:
        if document:
            combined_content = ""
            for file in document:
                if file.name.endswith(".pdf"):
                    combined_content += read_pdf(file)
                elif file.name.endswith(".txt"):
                    combined_content += read_txt(file)

            split = split_doc(combined_content, chunk_size, chunk_overlap)
            create_new_vs = existing_vector_store == "<New>"

            try:
                embedding_storing(split, create_new_vs, existing_vector_store, new_vs_name)
                st.success("Vector store saved successfully!")
            except ValueError as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please upload at least one file.")

def display_chatbot_page():
    st.title("Multi Source Chatbot")

    with st.expander("Initialize the LLM Model"):
        st.markdown("""
            Insert the token and select the vector store, temperature, and maximum character length to create the chatbot.
            **NOTE:** Token: API Key from Hugging Face. Temperature: How creative should the chatbot be? (Value between 0 and 1)
        """)

        with st.form("settings"):
            api_key = load_secrets()
            if api_key:
                token = st.text_input("Hugging Face Token (No need to insert)", type='password', value="******")
            else:
                token = st.text_input("Hugging Face Token (No need to insert)", type='password')
            llm_model = st.text_input("LLM Model", value="tiiuae/falcon-7b-instruct")
            instruct_embeddings = st.text_input("Instruct Embeddings", value="sentence-transformers/all-MiniLM-L6-v2")
            vector_store_list = os.listdir("vector store")
            default_choice = vector_store_list.index('naruto_snake') if 'naruto_snake' in vector_store_list else 0
            existing_vector_store = st.selectbox("Vector Store", vector_store_list, index=default_choice)
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            max_length = st.number_input("Maximum Character Length", value=200, min_value=1, step=1)
            create_chatbot = st.form_submit_button("Initialize Chatbot")

    if create_chatbot:
        if api_key:
            st.session_state.conversation = prepare_rag_llm(api_key, existing_vector_store, temperature, max_length)
            st.success("Chatbot initialized successfully!")
        else:
            st.error("Please enter a valid API key.")

    st.markdown("### Chat with the Bot")
    st.markdown("Enter your text below to get a response from the chatbot. **NOTE:** Initialize the LLM Model above before using the chatbot.")

    if 'conversation' in st.session_state:
        with st.form("chat", clear_on_submit=True):
            user_input = st.text_area("Your Question:", value="", height=150)
            submit_button = st.form_submit_button("Send")

        if submit_button and user_input:
            with st.spinner("Generating response..."):
                answer, doc_source = generate_answer(user_input, api_key)

            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**Chatbot:** {answer}")

            st.markdown("### Source Documents")
            for idx, source in enumerate(doc_source, 1):
                st.markdown(f"**{idx}.** {source}")
    else:
        st.warning("Please initialize the chatbot first.")

if __name__ == "__main__":
    main()
