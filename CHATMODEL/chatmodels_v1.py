""" IMPORTS AND ENV VAR """
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os 
from langchain_pinecone import PineconeVectorStore 
from pinecone import Pinecone, ServerlessSpec 
from dotenv import load_dotenv 
from langchain.schema import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda 
from langchain_core.output_parsers import StrOutputParser 
from langchain.prompts import PromptTemplate 
import streamlit as st 
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field 
from typing import Literal 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY2")
openai_api_key = os.getenv("OPENAI_API_KEY")
grok_api_key = os.getenv("GROK_API_KEY")
print(f"grok_api_key : {grok_api_key}")

""" VARIABLES """
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_hf = HuggingFaceEmbeddings(model_name=embedding_model_name)
grok_api = grok_api_key
llm = ChatOpenAI(
    model="grok-3-latest",
    temperature=0.2,
    max_tokens=None,
    max_retries=2,
    api_key=grok_api_key,
    base_url="https://api.x.ai/v1",
    http_client=None,
    default_headers={"Authorization": f"Bearer {grok_api}"}
)

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

# for i in os.listdir("./FILES"):
#     if i.endswith(".pdf"):
#         print(f"Loading {i}...")
#         loader = PyPDFLoader(f'./FILES/{i}')
#         doc = loader.load()
#         print(f"Splitting {i}...")
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
#         split = text_splitter.split_documents(doc)
#         print(f"Adding {i} to database...")
#         vector_store.add_documents(split)
#         print(f"{i} added to database !!!")

""" RETRIEVER SETUP """
classifier_llm = ChatOpenAI(
    model="grok-3-latest",
    temperature=0.2,
    max_tokens=None,
    max_retries=2,
    api_key=grok_api_key,
    base_url="https://api.x.ai/v1",
    http_client=None,
    default_headers={"Authorization": f"Bearer {grok_api}"}
)

class schema(BaseModel):
    category: Literal['non-medical question', 'medical question'] = Field(description="Identify if this a medical science related question or not")

prompt = PromptTemplate(
    input_variables=["query"],
    template="Identify if the statement is a medical science related question or not': {query}"
)

structured_llm = classifier_llm.with_structured_output(schema)
chain1 = prompt | structured_llm

medical_llm = ChatOpenAI(
    model="grok-3-latest",
    temperature=0.2,
    max_tokens=None,
    max_retries=2,
    api_key=grok_api_key,
    base_url="https://api.x.ai/v1",
    http_client=None,
    default_headers={"Authorization": f"Bearer {grok_api}"}
)

system_prompt = """You are a very professional technical receptionist. Answer queries based solely on the message history, without using any external sources. Respond to basic professional questions briefly. Define boundaries where a receptionist won't answer, limiting responses to medical science topics.
For example:
- Prompt: "what is the time today" Answer: Provide today's time.
- Prompt: "who is president of USA" Answer: I am sorry, I only discuss topics about medical science.
- Prompt: "what was my last question" Answer: Fetch the last user question from message history.
- Prompt: "what is your favourite movie" Answer: I am sorry, I only discuss topics about medical science."""

message_history = [SystemMessage(content=system_prompt)]

generic_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}")
])

while True:
    query = input("Enter a prompt : ")
    result = chain1.invoke({"query": query})
    message_history.append(HumanMessage(content=query))
    print(chain1.invoke(query).category)

    if result.category == "medical question":
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        arr = "\n\n".join(a.page_content for a in docs)
        medical_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Please find the context: {context} and question: {question} 
You are a document-based AI assistant. Your sole function is to provide information contained within the files provided in the Document store and its respective chunks. Confine your responses strictly to the data available in the documents. Avoid fetching data from external sources.

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
        )
        medical_chain = medical_prompt | medical_llm
        response = medical_chain.invoke({"context": arr, "question": query})
        print(response.content)
        message_history.append(AIMessage(content=response.content))
    else:
        generic_llm = ChatOpenAI(
            model="grok-3-latest",
            temperature=0.2,
            max_tokens=None,
            max_retries=2,
            api_key=grok_api_key,
            base_url="https://api.x.ai/v1",
            http_client=None,
            default_headers={"Authorization": f"Bearer {grok_api}"}
        )
        generic_chain = generic_prompt | generic_llm
        response = generic_chain.invoke({"history": message_history, "query": query})
        message_history.append(AIMessage(content=response.content))
        print(response.content)