"""  IMPORTS AND ENV VAR"""


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore 
from pinecone import Pinecone, ServerlessSpec 
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
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")
# grok_api_key = os.getenv("GROK_API_KEY")
# print(f"grok_api_key : {grok_api_key}")
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]
grok_api_key = st.secrets["GROK_API_KEY"]
print(f"grok_api_key : {grok_api_key}")


""" VARIABLES """
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_hf = HuggingFaceEmbeddings(model_name=embedding_model_name)

llm = ChatOpenAI(
    model="grok-3-latest",  # Use the model that worked in test.py
    temperature=0.2,
    max_tokens=None,
    max_retries=2,
    api_key=grok_api_key,  # xAI API key
    base_url="https://api.x.ai/v1",  # xAI API base URL
    http_client=None,  # Optional: can specify a custom client if needed
    default_headers={"Authorization": f"Bearer {grok_api_key}"}
)

# """  PDF LOADER"""

# loader = PyPDFLoader('./29_jan_morning.pdf')

# doc= loader.load()


# """ SPLITTING DOCUMENTS INTO TEXTS"""
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
# split = text_splitter.split_documents(doc)
# # print(f"Type of split : {type(split)}")
# print(f"Length of split : {len(split)}")
# print(f"First element of split : {split[0]}")
# print(f"Last element of split : {split[-1]}")


""" VECTOR DATABASE SETUP """
# Validate API keys


# Initialize Pinecone client (optional, only if you need to manage indexes directly)
pc = Pinecone(api_key=pinecone_api_key)
# embedding_dimension = 784
index_name = "rhl-project-2"
# embedding_dimension = len(embedding_hf.embed_query("Hello world"))
# pc.create_index(
#     name=index_name,
#     metric="cosine",
#     dimension=embedding_dimension,
#     spec=ServerlessSpec(cloud="aws", region="us-east-1")
# )


vector_store = PineconeVectorStore( index_name=index_name, embedding=embedding_hf, pinecone_api_key=pinecone_api_key )
print("connection to Pinecone Established !!!!")

""" UPLOADING DOCUMENTS TO VECTOR DATABASE """

# vector_store.add_documents(split)
# print("data added to database !!!")

# """ MODEL AND RETREIVER"""

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


arr=retriever.invoke('what are indications for Continuous positive airway pressure?')
# for i in arr:
#     print(i.page_content)
#     print()

# """ CHAINS"""


prompt = PromptTemplate(
    input_variables=["context","question"],
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

# while(True):
#     query = input("Enter your query : ")
#     result = chain.invoke(query)
#     print(result)

# """ UI  """


st.header("LangChain HUGGINGFACE(sentence-transformers/all-MiniLM-L6-v2) EMBEDDER GROK Demo")
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

