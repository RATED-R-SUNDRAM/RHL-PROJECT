""" IMPORTS AND ENV VAR """
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_pinecone import PineconeVectorStore 
from pinecone import Pinecone, ServerlessSpec 
from dotenv import load_dotenv 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_core.runnables import RunnableBranch, RunnablePassthrough 
from langchain.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field 
from typing import Literal 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
import os

load_dotenv()
# pinecone_api_key = st.secrets["PINECONE_API_KEY2"]
# openai_api_key = st.secrets["OPENAI_API_KEY"]
# grok_api_key = st.secrets["GROK_API_KEY"]
# print(f"grok_api_key : {grok_api_key}")


pinecone_api_key = os.getenv("PINECONE_API_KEY2")
openai_api_key = os.getenv("OPENAI_API_KEY")
grok_api_key = os.getenv("GROK_API_KEY")
print(f"grok_api_key : {grok_api_key}")

""" VARIABLES AND LLM SETUP """
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_hf = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Define all LLMs once
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

classifier_llm = ChatOpenAI(**base_llm_config)
medical_llm = ChatOpenAI(**base_llm_config)
generic_llm = ChatOpenAI(**base_llm_config)

""" VECTOR DATABASE SETUP """
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
print("Connection to Pinecone Established !!!!")

""" CLASSIFIER SETUP """
class schema(BaseModel):
    category: Literal['non-medical question', 'medical question'] = Field(description="Identify if this a medical science related question or not")

classifier_prompt = PromptTemplate(
    input_variables=["query"],
    template="Identify if the statement is a medical science related question or not': {query}"
)

structured_llm = classifier_llm.with_structured_output(schema)
classifier_chain = classifier_prompt | structured_llm

""" GENERIC LLM SETUP """
system_prompt = """You are a very professional technical receptionist. Answer queries based solely on the message history, without using any external sources. Respond to basic professional questions briefly. Define boundaries where a receptionist won't answer, limiting responses to medical science topics.
For example:
- Prompt: "what is the time today" Answer: Provide today's time.
- Prompt: "who is president of USA" Answer: I am sorry, I only discuss topics about medical science.
- Prompt: "what was my last question" Answer: Fetch the last user question from message history.
- Prompt: "what is your favourite movie" Answer: I am sorry, I only discuss topics about medical science."""

# Initialize session state variables at the top
""" IMPORTS AND ENV VAR """
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore 
from pinecone import Pinecone, ServerlessSpec 
from dotenv import load_dotenv 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_core.runnables import RunnableBranch, RunnablePassthrough 
from langchain.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field 
from typing import Literal 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
import os

load_dotenv()
# pinecone_api_key = st.secrets["PINECONE_API_KEY2"]
# openai_api_key = st.secrets["OPENAI_API_KEY"]
# grok_api_key = st.secrets["GROK_API_KEY"]
# print(f"grok_api_key : {grok_api_key}")
pinecone_api_key = os.getenv("PINECONE_API_KEY2")
openai_api_key = os.getenv("OPENAI_API_KEY")
grok_api_key = os.getenv("GROK_API_KEY")
print(f"grok_api_key : {grok_api_key}")

# Initialize session state variables at the top
if "message_history" not in st.session_state:
    system_prompt = """You are a very professional technical receptionist. Answer queries based solely on the message history, without using any external sources. Respond to basic professional questions briefly. Define boundaries where a receptionist won't answer, limiting responses to medical science topics.
    For example:
    - Prompt: "what is the time today" Answer: Provide today's time.
    - Prompt: "who is president of USA" Answer: I am sorry, I only discuss topics about medical science.
    - Prompt: "what was my last question" Answer: Fetch the last user question from message history.
    - Prompt: "what is your favourite movie" Answer: I am sorry, I only discuss topics about medical science."""
    st.session_state.message_history = [SystemMessage(content=system_prompt)]

if "medical_message_history" not in st.session_state:
    medical_system_prompt = """You are a document-based AI assistant. Your sole function is to provide information contained within the files provided in the Document store and its respective chunks. Confine your responses strictly to the data available in the documents. Avoid fetching data from external sources.

    Response Instructions:
    1. If a query cannot be answered based on the information in the documents, respond clearly that it is out of scope. Do not generate content or search for data beyond the provided documents.
    1.1 Never in any circumstance refer to any other source other than the content provided in the documents.
    1.2 Answer as if you're a professional medical advisor, base your response on the information provided in the documents but dont mention this in the answer.
    2. When the query is relevant but unclear, ask up to two follow-up questions, offering a maximum of three concise options (5 words or less each) only if necessary, preferably skip if not useful.
    3. Paraphrase, summarize, and offer different response formats while maintaining factual accuracy.
    4. Limit responses to 150 words or fewer. Use a helpful and compassionate tone.
    5. Handle greetings appropriately.
    6. Answer "Yes/No" or in a "Small paragraph" format based on the user's request. If unable to understand the query after two attempts, respond with "null."
    7. Do not use special characters while providing the content."""
    st.session_state.medical_message_history = [SystemMessage(content=medical_system_prompt)]

if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Rest of your code (LLM setup, vector store, etc.) follows...

generic_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}")
])

generic_chain = generic_prompt | generic_llm

""" MEDICAL LLM SETUP """
def get_context(query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    return "\n\n".join(a.page_content for a in docs)

# Update medical prompt to include message history
medical_prompt = ChatPromptTemplate.from_messages([
    ("system", st.session_state.medical_message_history[0].content),
    MessagesPlaceholder(variable_name="medical_history"),
    ("human", "Context: {context}\nQuestion: {question}")
])

medical_chain = (
    RunnablePassthrough.assign(context=lambda x: get_context(x["question"]))
    | RunnablePassthrough.assign(medical_history=lambda _: st.session_state.medical_message_history)
    | medical_prompt
    | medical_llm
)

""" COMBINED CHAIN WITH RUNNABLE BRANCH """
combined_chain = RunnableBranch(
    (lambda x: x["category"] == "medical question", medical_chain),
    generic_chain
)

""" STREAMLIT UI """
st.title("Medical Chatbot")
st.write("Ask your medical or general questions below. Chat history is preserved!")

# Display chat history
for message in st.session_state.message_history[1:]:  # Skip system message
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# Initialize input key in session state
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Input field for user query
query = st.text_input("Enter your query:", key=f"user_input_{st.session_state.input_key}")

if query:
    # Append user query to main history
    st.session_state.message_history.append(HumanMessage(content=query))
    
    # Classify the query
    category = classifier_chain.invoke({"query": query}).category
    
    if category == "medical question":
        # Append user query to medical message history
        st.session_state.medical_message_history.append(HumanMessage(content=query))
        
        # Get response from medical chain
        response = combined_chain.invoke({
            "category": category,
            "history": st.session_state.message_history,
            "query": query,
            "question": query
        })
        
        # Append response to medical message history
        st.session_state.medical_message_history.append(AIMessage(content=response.content))
        
        # Limit medical message history to last 4 messages (2 exchanges) to avoid bloat
        if len(st.session_state.medical_message_history) > 5:  # System + 2 exchanges (4 messages)
            st.session_state.medical_message_history = [st.session_state.medical_message_history[0]] + st.session_state.medical_message_history[-4:]
    else:
        # Handle generic chain as before
        response = combined_chain.invoke({
            "category": category,
            "history": st.session_state.message_history,
            "query": query,
            "question": query
        })
    
    # Append response to main history
    st.session_state.message_history.append(AIMessage(content=response.content))
    
    # Increment input key to reset input field
    st.session_state.input_key += 1
    
    # Refresh the page to display the new message
    st.rerun()