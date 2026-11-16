import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Page Configuration
st.set_page_config(page_title="AI Research Paper Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Research Paper Assistant (RAG + LLM + GenAI)")
st.write("Upload your research paper PDF, and let the AI help you understand, summarize, and explore it quickly!")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload Research Paper PDF", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_paper.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Process and Chunk the PDF
    if st.button("Process Research Paper"):
        with st.spinner("Reading, chunking, and indexing the paper... Please wait."):
            loader = PyPDFLoader("temp_paper.pdf")
            docs = loader.load()
            
            # Optimized chunk size for context preservation
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = text_splitter.split_documents(docs)

            # 3. Create Embeddings using a free local model
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # 4. Store in ChromaDB vector database
            vector_store = Chroma.from_documents(chunks, embeddings, collection_name="research_paper")
            
            # Save into session state
            st.session_state["vector_store"] = vector_store
            st.success("✅ Paper processed and indexed successfully! You can now ask questions below.")

# 5. Interactive Assistant Interface
if "vector_store" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Chat with Your Research Paper")
    
    # Quick action buttons for your project requirements
    
    
    
    
    
    
    
    
    
    
    

    # Text input for custom user questions
    user_query = st.text_input("Or type your custom question here:")

    if user_query:
        with st.spinner("🤖 AI is generating your answer..."):
            # Retrieval Step (RAG)
            retriever = st.session_state["vector_store"].as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.invoke(user_query)
            
            # Combine retrieved text chunks into a context string
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Generative AI Step (LLM)
            model_name = "google/flan-t5-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Prompt Engineering to guide the LLM like ChatGPT/Gemini
            input_text = f"Based strictly on the context provided below, answer the user's question clearly and professionally.\n\nContext:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"
            
            # Tokenize and generate response
            inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=200, num_beams=3, early_stopping=True)
            final_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        # Display AI Response
        st.success("### 💡 AI Assistant Response:")
        st.write(final_answer)
        
        # Expandable section to show raw source context for transparency
        with st.expander("🔍 View Retrieved PDF Source Chunks"):
            for i, doc in enumerate(relevant_docs):
                st.markdown(f"*Chunk {i+1}:*")
                st.write(doc.page_content)
                st.markdown("---")