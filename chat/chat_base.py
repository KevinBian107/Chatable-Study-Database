'''This need openai==0.28'''

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
import ast
import os
import logging

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
    return embedding_df, embeddings

def initialize_faiss(embeddings):
    logging.info("Initializing FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    logging.info("Adding embeddings to FAISS index...")
    index.add(embeddings)
    logging.info(f"FAISS index initialized with {index.ntotal} vectors.")
    return index

def get_similar_chunks(query, model, index, embedding_df, top_k=5):
    logging.info(f"Encoding the query: '{query}'")
    query_embedding = model.encode(query, convert_to_tensor=False).astype('float32')
    query_embedding = np.expand_dims(query_embedding, axis=0)
    logging.info("Searching for similar chunks in FAISS index...")
    distances, indices = index.search(query_embedding, top_k)
    similar_chunks = embedding_df.iloc[indices[0]]['chunk'].tolist()
    logging.info(f"Retrieved {len(similar_chunks)} similar chunks.")
    return similar_chunks

def get_similar_chunks_by_quarter(query, model, index, embedding_df, top_k=5, quarter=None):
    logging.info(f"Encoding the query: '{query}'")
    query_embedding = model.encode(query, convert_to_tensor=False).astype('float32')
    query_embedding = np.expand_dims(query_embedding, axis=0)
    logging.info("Searching for similar chunks in FAISS index...")
    distances, indices = index.search(query_embedding, top_k)
    
    # If quarter is specified, filter the dataframe to include only relevant rows
    if quarter:
        logging.info(f"Filtering data for quarter: {quarter}")
        relevant_chunks = embedding_df[embedding_df['quarter'] == quarter]
        if relevant_chunks.empty:
            logging.warning(f"No data found for quarter: {quarter}")
            return []
    else:
        relevant_chunks = embedding_df
    
    # Ensure the number of indices does not exceed the number of relevant rows
    num_relevant_rows = relevant_chunks.shape[0]
    top_k = min(top_k, num_relevant_rows)
    
    # Adjust indices if they are out-of-bounds
    valid_indices = [idx for idx in indices[0] if idx < num_relevant_rows]
    
    if not valid_indices:
        logging.warning(f"No valid indices found for the given query.")
        return []
    
    relevant_chunks = relevant_chunks.iloc[valid_indices]
    similar_chunks = relevant_chunks['chunk'].tolist()
    
    logging.info(f"Retrieved {len(similar_chunks)} similar chunks.")
    return similar_chunks


def generate_response(prompt, context_chunks, temperature=0.3, top_p=0.85):
    logging.info("Generating response from OpenAI...")
    context = "\n\n".join(context_chunks)
    combined_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer (use only information from the context):"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": combined_prompt}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature, 
            top_p=top_p,
            max_tokens=500,
            n=1,
            stop=None
        )
        answer = response['choices'][0]['message']['content'].strip()
        logging.info("Response generated successfully.")
        return answer
    except Exception as e:
        logging.error(f"An error occurred while generating the response: {e}")
        return "Sorry, I couldn't generate a response at this time."


def main():
    logging.info("Starting the Study Assistant...")
    
    # Load and initialize
    embedding_df, embeddings = load_embeddings('embeddings.csv')
    index = initialize_faiss(embeddings)
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Ensure consistency
    
    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logging.critical("OpenAI API Key is not set. Please set the OPENAI_API_KEY environment variable.")
        return
    
    logging.info("Study Assistant is ready to receive your questions.")
    
    while True:
        user_prompt = input("\nPlease ask from 2022 Fall forward\nYour Question (type 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            logging.info("Goodbye!")
            break
        similar_chunks = get_similar_chunks_by_quarter(user_prompt, model, index, embedding_df, top_k=5, quarter='2022Q4')
        answer = generate_response(user_prompt, similar_chunks)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()