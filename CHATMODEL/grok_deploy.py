from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore 
from pinecone import Pinecone, ServerlessSpec 
from dotenv import load_dotenv 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_core.runnables import RunnablePassthrough 
from langchain.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import streamlit as st
import os

# Load environment variables
load_dotenv()
pinecone_api_key = st.secrets["PINECONE_API_KEY2"]
openai_api_key = st.secrets["OPENAI_API_KEY"]
grok_api_key = st.secrets["GROK_API_KEY"]
# pinecone_api_key = os.getenv("PINECONE_API_KEY2")
# grok_api_key = os.getenv("GROK_API_KEY")

# Embedding model setup
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_hf = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": "cpu"})

# LLM configuration
base_llm_config = {
    "model": "grok-3-latest",
    "temperature": 0.2,
    "max_tokens": None,
    "max_retries": 2,
    "api_key": grok_api_key,
    "base_url": "https://api.x.ai/v1",
    "http_client": None,
    "default_headers": {"Authorization": f"Bearer {grok_api_key}"}
}

medical_llm = ChatOpenAI(**base_llm_config)

# Vector DB Setup
pc = Pinecone(api_key=pinecone_api_key)
embedding_dimension = len(embedding_hf.embed_query("test"))
index_name = "chat-models-v1-all-minilm-l6"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        metric="cosine",
        dimension=embedding_dimension,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

vector_store = PineconeVectorStore(index_name=index_name, embedding=embedding_hf, pinecone_api_key=pinecone_api_key)

# Prompt Template
medical_prompt = PromptTemplate(
    input_variables=["context", "question", "word_limit"],
    template="""Please find the context: {context} and question: {question} 
You are a document-based AI assistant. Your sole function is to provide information contained within the files provided in the Document store and its respective chunks. Confine your responses strictly to the data available in the documents. Avoid fetching data from external sources.

Response Instructions:
1. If a query cannot be answered based on the information in the documents, respond clearly that it is out of scope. Do not generate content or search for data beyond the provided documents.
1.1 Never in any circumstance refer to any other source other than the content provided in the documents.
1.2 Answer as if you're a professional medical advisor, base your response on the information provided in the documents but dont mention this in the answer.
2. When the query is relevant but unclear, ask up to two follow-up questions, offering a maximum of three concise options (5 words or less each) only if necessary, preferably skip if not useful.
3. Paraphrase, summarize, and offer different response formats while maintaining factual accuracy.
4. Strictly Limit responses to {word_limit} words or fewer. Use a helpful and compassionate tone.
5. Handle greetings appropriately.
6. Answer "Yes/No" or in a "Small paragraph" format based on the user's request. If unable to understand the query after two attempts, respond with "null."
7. Do not use special characters while providing the content."""
)

def get_context(query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    return "\n\n".join(a.page_content for a in docs)

# Streamlit UI
st.title("Medical Chatbot")
st.write("Ask your medical questions below. Chat history is preserved!")

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "input_key" not in st.session_state:
    st.session_state.input_key = 0

query = st.text_input("Enter your medical query:", key=f"user_input_{st.session_state.input_key}")
word_limit = st.text_input("Response word limit:", value="100", key=f"word_limit_{st.session_state.input_key}")

if query and word_limit:
    st.session_state.message_history.append(HumanMessage(content=query))
    response = (
        RunnablePassthrough.assign(context=lambda x: get_context(x["question"]))
        | medical_prompt
        | medical_llm
    ).invoke({
        "question": query,
        "word_limit": word_limit
    })
    st.session_state.message_history.append(AIMessage(content=response.content))
    st.session_state.input_key += 1
    st.rerun()

# Show chat
for message in st.session_state.message_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)
