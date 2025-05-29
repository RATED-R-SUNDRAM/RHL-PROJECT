""" IMPORTS AND ENV VAR """
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # Changed to Chroma
import os 
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

load_dotenv()
# Removed pinecone_api_key since Chroma doesn't need it
openai_api_key = os.getenv("OPENAI_API_KEY")
grok_api_key = os.getenv("GROK_API_KEY")
print(f"grok_api_key : {grok_api_key}")

""" VARIABLES """
embedding_model_name = "jinaai/jina-embeddings-v2-small-en"
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

# """ PDF LOADER """
# loader = PyPDFLoader('./29_jan_morning.pdf')
# doc = loader.load()

# """ SPLITTING DOCUMENTS INTO TEXTS """
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
# split = text_splitter.split_documents(doc)

""" VECTOR DATABASE SETUP """
# Removed Pinecone setup (no index creation or API key needed)
# Chroma stores embeddings locally in the specified directory
persist_directory = "./chroma_db"  # Directory to store the Chroma database

persist_directory = "./chroma_db"
collection_name = "rhl-project-jina2"

# Initialize or load Chroma vector store
if os.path.exists(persist_directory):
    print("Loading existing Chroma vector store...")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_hf,
        collection_name=collection_name,
    )
else:
    print("Creating new Chroma vector store...")
    vector_store = Chroma.from_documents(
        documents=split,
        embedding=embedding_hf,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
print("Chroma vector store initialized and documents added !!!!")

# Persist the database to disk
vector_store.persist()

# """ MODEL AND RETRIEVER """
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

arr = retriever.invoke('what are indications for Continuous positive airway pressure?')

""" CHAINS """
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="You are a helpful assistant answering an exam question. Use only this information: {context}\nDo not use external knowledge. Do not mention, describe, or reference the information's content, focus, or origin in your response. Answer this question in ~100 words only if the information is directly relevant: `{question}`\nIf the question is unrelated, reply only: 'I have no information on this topic'. Do not explain why the question is unrelated.""You are a helpful assistant answering an exam question. Use only this information: {context}\nDo not use any external knowledge or mention where the information comes from. Answer this question in ~100 words if the information is relevant: `{question}`\nIf the question is unrelated, reply only: 'I have no information on this topic'."
)
parser = StrOutputParser()

def agg_func(docs):
    return "/n/n".join(a.page_content for a in docs)

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(agg_func),
    'question': RunnablePassthrough()
})

chain = parallel_chain | prompt | llm | parser

""" UI """
st.header("LangChain JINA(jinaai/jina-embeddings-v2-small-en) EMBEDDER GROK Demo")
st.markdown("""
Welcome! This tool answers medical-related questions using context from the uploaded PDF.
Just type your question below and click *Submit*.
""")
user_input = st.text_input("Enter a prompt")

if st.button("Generate"):
    response = chain.invoke(user_input)
    st.write(f"RESPONSE : {response}")
    st.write()
    st.write()
    st.write(f"========THE RELEVANT TEXT RETRIEVED FOR THIS PROMPTS ARE ===================:")
    st.write(agg_func(retriever.invoke(user_input)))

