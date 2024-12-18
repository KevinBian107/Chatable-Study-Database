'''This need to have openai>=1.40.0'''

import os
import logging
import pandas as pd
import numpy as np
import faiss
import ast

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import ChatPromptTemplate

from pathlib import Path
import sys

main_path = Path(__file__).resolve().parent.parent.parent.parent
if str(main_path) not in sys.path:
    sys.path.append(str(main_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def load_embeddings(file_path):
    logging.info("Loading embeddings from CSV...")
    embedding_df = pd.read_csv(file_path)
    logging.info("Parsing embedding strings into lists...")
    embedding_df['embedding'] = embedding_df['embedding'].apply(ast.literal_eval)
    logging.info("Converting embeddings to NumPy array...")
    embeddings = np.array(embedding_df['embedding'].tolist()).astype('float32')
    logging.info("Embeddings loaded successfully.")
    return embedding_df

def initialize_faiss_vector_store(embedding_df):
    logging.info("Initializing FAISS vector store with LangChain...")
    texts = embedding_df['chunk'].tolist()
    
    # Ensure 'source' is in metadata
    if 'source' not in embedding_df.columns:
        embedding_df['source'] = 'unknown'  # Customize as needed
    
    metadatas = embedding_df.drop(['embedding', 'chunk'], axis=1).to_dict(orient='records')
    
    # Initialize embeddings
    embedding_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # Create FAISS vector store
    vector_store = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
    logging.info(f"FAISS vector store initialized with {vector_store.index.ntotal} vectors.")
    return vector_store

def create_retrieval_chain(llm, vector_store, search_kwargs=None):
    """
    Creates a RetrievalQAWithSourcesChain.

    Args:
        llm: The language model instance.
        vector_store: The FAISS vector store.
        search_kwargs: Dictionary of search parameters, including filters.

    Returns:
        RetrievalQAWithSourcesChain instance.
    """
    logging.info("Setting up the retrieval QA chain with LangChain...")
    
    # Create a retriever with optional search_kwargs
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs or {"k": 5})
    
    # Optionally, define a custom prompt template if needed
    # For simplicity, we'll use the default prompt provided by RetrievalQAWithSourcesChain
    
    # Create RetrievalQAWithSourcesChain
    qa_chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=True
    )
    
    logging.info("Retrieval QA chain setup complete.")
    return qa_chain

def main():
    logging.info("Starting the Study Assistant with LangChain...")
    
    # Verify OpenAI API Key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logging.critical("OpenAI API Key is not set. Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Initialize OpenAI Chat Model
    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=openai_api_key
    )
    
    # Load embeddings
    embedding_df = load_embeddings('embeddings.csv')
    
    # Initialize FAISS vector store
    vector_store = initialize_faiss_vector_store(embedding_df)
    
    # Create the initial RetrievalQA chain without filters
    qa_chain = create_retrieval_chain(llm=chat, vector_store=vector_store, search_kwargs={"k": 5})
    
    logging.info("Study Assistant is ready to receive your questions.")
    
    while True:
        user_prompt = input("\nPlease ask from 2022 Fall forward\nYour Question (type 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            logging.info("Goodbye!")
            break
        
        # Optionally, prompt for quarter filtering
        quarter = input("Enter the quarter (e.g., '2022Q4') or press Enter to skip: ").strip()
        
        if quarter:
            logging.info(f"Filtering data for quarter: {quarter}")
            # Define search_kwargs with filter
            search_kwargs = {
                "k": 5,
                "filter": {"quarter": quarter}
            }
            # Create a new QA chain with the filtered retriever
            filtered_qa_chain = create_retrieval_chain(llm=chat, vector_store=vector_store, search_kwargs=search_kwargs)
            current_qa_chain = filtered_qa_chain
        else:
            current_qa_chain = qa_chain
        
        # Execute the QA chain using `invoke` instead of `run`
        try:
            # Invoke the chain and capture all outputs
            result = current_qa_chain.invoke({"question": user_prompt})
            # Access the 'answer' key from the result
            answer = result.get('answer', 'No answer found.')
            # Optionally, access 'sources' if needed
            sources = result.get('sources', [])
            
            print(f"\nAnswer: {answer}")
            if sources:
                print("\nSources:")
                for source in sources:
                    print(f"- {source}")
        except Exception as e:
            logging.error(f"An error occurred while generating the response: {e}")
            print("Sorry, I couldn't generate a response at this time.")

if __name__ == "__main__":
    main()
