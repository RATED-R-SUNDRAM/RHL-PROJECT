#%%
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import _answer_relevance, faithfulness, context_precision, context_recall
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore 
from pinecone import Pinecone, ServerlessSpec 
import os 
from langchain_openai import OpenAIEmbeddings
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

# Step 1: Load the dataset
df = pd.read_excel("Questions_answers_comparision.xlsx",sheet_name='RESPONSE')
print(f"Loaded dataset with {len(df)} prompts.")
df = df.iloc[:,[0,2,3,4,5,6,7,8,9]]
print(df.columns)

df.columns=['PROMPTS', 'GROUND_TRUTH_RESPONSE','OPENAI_EMBEDDER_OPENAI_LLM', 'OPENAI_EMBEDDER_GROK_LLM',
       'OLLAMA-NOMIC-EMBEDDER-GROk-LLM', 'HUGGINGFACE - All-mini-l6m-GROK-LLM',
       'MIXBREAD_AI-EMBEDDER-GROK-LLM', 'BAAI/bge_large-EMBEDDER-GROK-LLM',
       'BAAI/bge_small-EMBEDDER-GROK-LLM']
print(df.columns)

# Step 2: Define embedders for each model
models=['OPENAI_EMBEDDER_OPENAI_LLM', 'OPENAI_EMBEDDER_GROK_LLM',
       'OLLAMA-NOMIC-EMBEDDER-GROk-LLM', 'HUGGINGFACE - All-mini-l6m-GROK-LLM',
       'MIXBREAD_AI-EMBEDDER-GROK-LLM', 'BAAI/bge_large-EMBEDDER-GROK-LLM',
       'BAAI/bge_small-EMBEDDER-GROK-LLM']

# Step 3: Generate contexts and prepare RAGAS data for each model
eval_datasets = {}
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_api_key2 = os.getenv("PINECONE_API_KEY2")
openai_api_key = os.getenv("OPENAI_API_KEY")
grok_api_key = os.getenv("GROK_API_KEY")


embedders ={
    'OPENAI_EMBEDDER_OPENAI_LLM': [OpenAIEmbeddings(model =  'text-embedding-3-large',dimensions= 784, openai_api_key=os.getenv("OPENAI_API_KEY")), 'openai-embedder-openai-llm', pinecone_api_key],
    'OPENAI_EMBEDDER_GROK_LLM': [OpenAIEmbeddings(model =  'text-embedding-3-large',dimensions= 784, openai_api_key=os.getenv("OPENAI_API_KEY")), 'openai-embedder-openai-llm', pinecone_api_key],
    'OLLAMA-NOMIC-EMBEDDER-GROk-LLM': [HuggingFaceEmbeddings( model_name='nomic-ai/nomic-embed-text-v1',model_kwargs={"trust_remote_code": True}), "ollama-nomic-embedder-grok-llm", pinecone_api_key],
    'HUGGINGFACE - All-mini-l6m-GROK-LLM': [HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'), 'chat-models-v1-all-minilm-l6', pinecone_api_key2],
    'MIXBREAD_AI-EMBEDDER-GROK-LLM': [HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1'), 'mixbread-embedder-grok-llm', pinecone_api_key2],
    'BAAI/bge_large-EMBEDDER-GROK-LLM': [HuggingFaceEmbeddings(model_name='BAAI/bge-large-en-v1.5'), 'rhl-project-baii-large', pinecone_api_key],
    'BAAI/bge_small-EMBEDDER-GROK-LLM': [HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5'), 'bge-small', pinecone_api_key2]
}

for model_name in models:
    print(f"Preparing data for {model_name}...")
    
    # Initialize the embedder for this model
    embedding_hf = embedders[model_name][0]
    
    # Update the vector store with the new embedder
    # Note: Assumes Pinecone index is already set up with documents for each embedder
    vector_store = PineconeVectorStore(
        index_name=embedders[model_name][1],
        embedding=embedding_hf,
        pinecone_api_key=embedders[model_name][2]
    )
    
    # Create the retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    # Function to get contexts
    def get_context(query):
        docs = retriever.invoke(query)
        return [doc.page_content for doc in docs]  # List of contexts for RAGAS
    
    # Prepare RAGAS data for this model
    eval_data = []
    for _, row in df.iterrows():
        eval_data.append({
            "question": row["PROMPTS"],
            "ground_truth": row["GROUND_TRUTH_RESPONSE"],
            "answer": row[model_name],
            "contexts": get_context(row["PROMPTS"])
        })
    
    eval_datasets[model_name] = eval_data

# Step 4: Evaluate each model with RAGAS
# Step 4: Evaluate each model with RAGAS
results = {}
for model_name, eval_data in eval_datasets.items():
    print(f"\nEvaluating {model_name} with RAGAS...")
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(eval_data)
    
    # Run evaluation (fix: use answer_relevance instead of _answer_relevance)
    result = evaluate(
        dataset=dataset,
        metrics=[_answer_relevance, faithfulness, context_precision, context_recall],  # Fixed here
    )
    
    results[model_name] = result
    print(f"Results for {model_name}: {result}")

# Step 5: Summarize results
print("\nSummary of RAGAS Metrics Across All Prompts:")
metrics = ["_answer_relevance", "faithfulness", "context_precision", "context_recall"]
summary_df = pd.DataFrame({model: {metric: results[model][metric] for metric in metrics} for model in results})
print(summary_df)
summary_df.to_csv("summary_df.csv",index=False)
# %%
