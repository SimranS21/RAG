import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
import torch
from langchain.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from utils.audio_utils import extract_audio_text
from utils.video_utils import extract_video_text

from utils.document_loaders import (
    process_logs,
    load_text_documents,
    load_text_documents,
    load_word_documents,
    load_pdf_documents,
)

# Function to clear CUDA memory
def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # st.write("CUDA memory cleared successfully.")

# Intent classification chain (cached for performance)
def get_intent_chain():
    intent_prompt = PromptTemplate(
        input_variables=["query"],
        template="""
You are an assistant that classifies user input into two categories: "document_query" or "chitchat".

Classify the following input:
"{query}"

Only respond with one of the following: document_query or chitchat.
"""
    )
    return LLMChain(llm=Ollama(model="llama3.2:latest", temperature=0), prompt=intent_prompt)


# Initialize device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Clear CUDA memory before processing
clear_cuda_memory()

# Initialize Milvus client
client = MilvusClient("milvus_database.db")
client.create_collection(
    collection_name="my_collection",
    dimension=768
)

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def chunk_documents(data, chunk_size=1000, chunk_overlap=200):
    """Split documents into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(data)
    # for doc in chunks:
    #     doc.metadata.setdefault("keywords", "")
    return chunks


def process_and_store_documents(documents):
    """Process and store documents in Milvus."""
    clear_cuda_memory()  # Clear memory before processing each batch
    chunks = chunk_documents(documents)
    st.write("Chunks generated:")
    st.write(chunks)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,  # Pass the embeddings
        connection_args={"uri": "./milvus_database.db"},
        drop_old=True,
    )

    st.success("Documents processed and stored in Milvus successfully!")

def app():
    st.title("🚀Multimodal Agentic AI Integration")
    st.subheader("📂 Upload your files or query the VectorDB")

    # Switch between upload and query mode
    mode = st.radio("Choose mode", ["Upload Files", "Query"])

    if mode == "Upload Files":
        uploaded_file = st.file_uploader("Choose a file",
                                         type=["mp3", "mp4", "wav", "txt", "csv", "yaml", "json", "docx", "pdf"])

        if st.button("Process File"):
            with st.spinner("⏳ Processing file... Please wait."):
                if uploaded_file:
                    file_type = uploaded_file.type
                    st.write(f"Detected file type: {file_type}")

                    documents = None

                    if "audio" in file_type:
                        st.audio(uploaded_file, format="audio/wav")
                        st.write("🎵Processing audio...")
                        text = extract_audio_text(uploaded_file)
                        documents = [
                            Document(page_content=text, metadata={"source": "audio", "file_name": uploaded_file.name,"keywords": ""})]

                    elif "video" in file_type:
                        st.video(uploaded_file)
                        st.write("Processing video...")
                        text = extract_video_text(uploaded_file)
                        documents = [
                            Document(page_content=text, metadata={"source": "video", "file_name": uploaded_file.name, "keywords": ""})]

                    elif "csv" in file_type or "yaml" in file_type or "json" in file_type:
                        st.write("Processing structured logs...")
                        documents = process_logs(uploaded_file, file_type, uploaded_file.name)

                    elif "document" in file_type:
                        st.write("📄 Processing Word document...")
                        documents = load_word_documents(uploaded_file)

                    elif "pdf" in file_type:
                        st.write("📜 Processing PDF document...")
                        documents = load_pdf_documents(uploaded_file)

                    elif "text" in file_type:
                        st.write("📝 Processing text document...")
                        documents = load_text_documents(uploaded_file)

                    if documents:
                        st.write("✅ Processing complete.\n Storing in Milvus...")
                        process_and_store_documents(documents)
                    else:
                        st.error("⚠ Failed to process the document. Please check the file type.")

    elif mode == "Query":
        if "history" not in st.session_state:
            st.session_state.history = []

        st.subheader("🧠 Ask a Question")

        with st.form("chat_form", clear_on_submit=True):
            query = st.text_input("💬 Ask something:",
                                  key="query_input",
                                  placeholder="Ask anything",
                                  label_visibility="collapsed"
                                  )
            submitted = st.form_submit_button("📤 Search", use_container_width=True)

            if submitted and query:
                st.write(f"Searching for: {query}")
                clear_cuda_memory()  # Clear CUDA memory before querying

                embeddings = HuggingFaceEmbeddings()

                vectorstore = Milvus(
                    embedding_function=embeddings,
                    connection_args={"uri": "./milvus_database.db"},
                    collection_name="LangChainCollection",
                )

                retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1})
                docs = retriever.invoke(query)

                # st.write(docs)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                llm = Ollama(model="llama3.2:latest", temperature=0.1)

                # Intent detection
                def is_chitchat_heuristic(text):
                    smalltalk = ["thanks", "thank you", "cool", "hi", "hello", "bye", "goodbye"]
                    return any(phrase in text.lower() for phrase in smalltalk)

                intent_chain = get_intent_chain()
                if is_chitchat_heuristic(query):
                    res = llm.invoke(query)
                else:
                    intent = intent_chain.run(query).strip().lower()
                    if intent == "chitchat":
                        res = llm.invoke(query)
                    else:
                        # Prompt template
                        PROMPT_TEMPLATE = """
                        Human: You are an AI assistant that provides answers to questions using fact-based and statistical information when possible.
                        Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
                        If relevant information is found, use it to answer the question. If no matching information is found, use your general knowledge to provide a helpful response do not make incorrect assumption .
                        If the question is related to an uploaded document draw from that document's content.
                        If you truly do not know the answer, just say that you don't know—don't try to make one up.
                        Always answer in 1-2 lines.
                        
                        
                        {context}
            
                        {question}
            
                        The response should be specific and use statistics or numbers when possible.
            
                        Assistant:"""

                        prompt = PromptTemplate(
                            template=PROMPT_TEMPLATE, input_variables=["context", "question"]
                        )

                        # # Check what retriever returns
                        # retrieved_context = retriever.invoke(query)
                        # st.write("🔍 Retrieved Context:", retrieved_context)
                        #
                        # # Format context properly
                        # if not retrieved_context:
                        #     retrieved_context = "No relevant documents found."

                        rag_chain = (
                                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                                | prompt
                                | llm
                                | StrOutputParser()
                        )

                        # Invoke the RAG chain to generate a response
                        res = rag_chain.invoke(query)

                # Store to session history
                st.session_state.history.append((query, res))


                # Chat History Display (even when not submitting)
                if st.session_state.history:
                    for q, r in reversed(st.session_state.history):  # show latest on top
                        with st.chat_message("user"):
                            st.markdown(q)
                        with st.chat_message("assistant"):
                            st.markdown(r)

if __name__ == "__main__":
    app()
