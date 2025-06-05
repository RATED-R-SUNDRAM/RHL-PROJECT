""" IMPORTS AND ENV VAR """
from langchain_community.embeddings import OllamaEmbeddings  # Changed to OllamaEmbeddings
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

load_dotenv()
# Removed pinecone_api_key since Chroma doesn't need it
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
grok_api_key = os.getenv("GROK_API_KEY")
print(f"grok_api_key : {grok_api_key}")

""" VARIABLES """
embedding_model_name = "BAAI/bge-large-en-v1.5"
embedding_hf = HuggingFaceEmbeddings(model_name=embedding_model_name)
grok_api= grok_api_key
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
# print(f"Type of split : {type(split)}")
# print(f"Length of split : {len(split)}")

# """ VECTOR DATABASE SETUP """
pc = Pinecone(api_key=pinecone_api_key)
embedding_dimension = 1024  # Updated to match nomic-embed-text's dimension (previously 784)

index_name = "rhl-project-baii-large"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        metric="cosine",
        dimension=embedding_dimension,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

vector_store = PineconeVectorStore(index_name=index_name, embedding=embedding_hf, pinecone_api_key=pinecone_api_key)
print("Connection to Pinecone Established !!!!")

# Uncomment to load PDFs
# """ VECTOR DATABASE SETUP """

# for i in os.listdir("./FILES"):
#     if i.endswith(".pdf"):
#         print(f"Loading {i}...")
#         loader = PyPDFLoader(f'./FILES/{i}')
#         doc = loader.load()
#         print(f"Splitting {i}...")
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
#         split = text_splitter.split_documents(doc)
#         print(f"Adding {i} to database...")
        
#         # Batch the upload (e.g., 50 documents per batch)
#         batch_size = 50
#         for j in range(0, len(split), batch_size):
#             batch = split[j:j + batch_size]
#             vector_store.add_documents(batch)
#             print(f"Uploaded batch {j // batch_size + 1} for {i}")
        
#         print(f"{i} added to database !!!")


print("TEST 1")
# vector_store.add_documents(split)  # Uncomment to add documents
# print("data added to database !!!")

""" MODEL AND RETRIEVER """
print("TEST 2")
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

print("TEST 3")
# arr = retriever.invoke('what are indications for Continuous positive airway pressure?')
# print(f"arr", arr)
# for i in arr:
#     print(i.page_content)
#     print()

# """ CHAINS """
# print("TEST 4")

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Please find the context: {context} and question: {question} 
You are a document-based AI assistant. Your sole function is to provide information contained within the files provided in the Document store and its respective chunks. Confine your responses strictly to the data available in the documents. Avoid fetching data from external sources.

Response Instructions:
1. If a query cannot be answered based on the information in the documents, respond clearly that it is out of scope. Do not generate content or search for data beyond the provided documents.
1.1 Never in any circumstance refer to any other source other than the content provided in the documents.
1.2 Answer as if you're a professional medical advisor, base your response on the information provided in the documents but dont mention this in the answer.
2. When the query is relevant but unclear, ask up to two follow-up questions, offering a maximum of three concise options (5 words or less each) only if necessary, preferably skip if not useful.
3. Paraphrase, summarize, and offer different response formats while maintaining factual accuracy.
4. Limit responses to necessary word limit . Use a helpful and compassionate tone.
5. Handle greetings appropriately.
6. Answer "Yes/No" or in a "Small paragraph" format based on the user's request. If unable to understand the query after two attempts, respond with "null."
7. Do not use special characters while providing the content."""
)
parser = StrOutputParser()

def agg_func(docs):
    return "/n/n".join(a.page_content for a in docs)

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(agg_func),
    'question': RunnablePassthrough()
})

chain = parallel_chain | prompt | llm | parser

arr= ['Can you tell me the indications of magnesium sulfate administration in pregnant women for fetal neuroprotection?',
'Can you tell me the indications for CPAP in a newborn?',
'When is administration of antenatal corticosteroid therapy for pregnant women recommended? What is the importance of antenatal corticosteroid therapy?',
'What is the recommendation of plastic wrapping for prevention of hypothermia in preterm neonates?',
'What is the dose of vitamin K that should be given as essential newborn care at birth for preterm baby. I mean do we give the same dose of vitamin K for term and preterm neonates?',
'A lactating mother has one sided breast redness, pain, and swelling. Can she continue breastfeeding her newborn baby?',
'A 3 days old newborn presented with failure to suck the breast otherwise unremarkable. I was wondering if I should start antibiotics for suspected sepsis?',
'A 2 days old neonate who has yellowish discoloration of the skin brought to our hospital. How do I treat him?',
'My newborn baby has purulent discharge from the eyes. Is it serious? What should I do?',
'When should I bath my newborn baby?',
'A mother who gave birth 2 days ago has difficulty providing adequate amount of expressed breast milk to her newborn. How should I proceed on supporting and feeding the newborn?',
'In our setup, we don’t have a readymade 10% dextrose IV fluid; but we have 40% and 5% dextrose. How can I prepare the required amount of 10% dextrose from the available fluids']

# for i in arr:
#     result = chain.invoke(i)
    
#     print("==========================================")
#     print(f"QUERY IS {i}")
#     print()
#     print(f"RESPONSE IS {result}")
#     print()
#     print()
retriever2 = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
for i in arr :
    print("=======================================================================================")
    docs = retriever2.invoke(i)
    print(len(docs))
    print("/n".join(a.page_content for a in docs))
    # print("=====================================")
    # print(chain.invoke(i))





# """ UI """
# st.header("LangChain OLLAMA EMBEDDER GROk AI")
# st.markdown("""
# Welcome! This tool answers medical-related questions using context from the uploaded PDF.
# Just type your question below and click *Submit*.
# """)
# user_input = st.text_input("Enter a prompt")

# if st.button("Generate"):
#     response = chain.invoke(user_input)
#     st.write(f"RESPONSE : {response}")
#     st.write()
#     st.write()
#     st.write(f"========THE RELEVANT TEXT RETRIEVED FOR THIS PROMPTS ARE ===================:")
#     st.write(agg_func(retriever.invoke(user_input)))

