"""  IMPORTS AND ENV VAR"""

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter , RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import streamlit as st


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
embedding_oai = OpenAIEmbeddings(model =  'text-embedding-3-large',dimensions= 784, openai_api_key=os.getenv("OPENAI_API_KEY"))

"""  PDF LOADER"""

# loader = PyPDFLoader('./29_jan_morning.pdf')

# doc= loader.load()


""" SPLITTING DOCUMENTS INTO TEXTS"""
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
# split = text_splitter.split_documents(doc)
# print(f"Type of split : {type(split)}")
# print(f"Length of split : {len(split)}")
# print(f"First element of split : {split[0]}")
# print(f"Last element of split : {split[-1]}")


""" VECTOR DATABASE SETUP """
# Validate API keys
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set.")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Initialize Pinecone client (optional, only if you need to manage indexes directly)
#pc = Pinecone(api_key=pinecone_api_key)
# embedding_dimension = 784
index_name = "rhl-project"
# pc.create_index(
#     name=index_name,
#     metric="cosine",
#     dimension=embedding_dimension,
#     spec=ServerlessSpec(cloud="aws", region="us-east-1")
# )
# print(f"Index '{index_name}' created successfully.")

vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding_oai,
    pinecone_api_key=pinecone_api_key
)

print("connection to Pinecone Established !!!!")

""" UPLOADING DOCUMENTS TO VECTOR DATABASE """

# vector_store.add_documents(split)
# print("data added to database !!!")

""" MODEL AND RETREIVER"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

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

""" UI  """


st.header("LangChain OpenAI Demo")
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
