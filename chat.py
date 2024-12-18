'''This script requires openai==0.28'''

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
import ast
import os
import logging
import pickle
import time
from transformers import pipeline
from typing import List, Optional

# =======================
# Configuration Settings
# =======================

CONFIG = {
    "EMBEDDINGS_CSV": "embeddings/embeddings.csv",
    "FAISS_INDEX_PATH": "embeddings/faiss.index",
    "EMBEDDINGS_PICKLE": "embeddings/embeddings.pkl",
    "MODEL_NAME": "all-MiniLM-L6-v2",
    "SUMMARIZER_MODEL": "facebook/bart-large-cnn",
    "OPENAI_MODEL": "gpt-4",
    "TOP_K_DEFAULT": 5,
    "TOP_K_MAX": 10,
    "FAISS_NLIST": 100,  # Will be adjusted based on dataset size
    "LOG_LEVEL": "INFO",
    "MAX_CONTEXT_LENGTH": 2000,
    "SUMMARY_MAX_LENGTH": 1000,
    "SUMMARY_MIN_LENGTH": 500,
    "CACHE_ENABLED": True,
    "MAX_RETRIES": 5,
    "BACKOFF_FACTOR": 2,
}

# =======================
# Configure Logging
# =======================

logging.basicConfig(
    level=CONFIG["LOG_LEVEL"],  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# =======================
# Utility Functions
# =======================

def load_embeddings(file_path: str):
    logging.info("Loading embeddings from CSV...")
    embedding_df = pd.read_csv(file_path)
    logging.info("Parsing embedding strings into lists...")
    embedding_df['embedding'] = embedding_df['embedding'].apply(ast.literal_eval)
    logging.info("Converting embeddings to NumPy array...")
    embeddings = np.array(embedding_df['embedding'].tolist()).astype('float32')
    logging.info("Embeddings loaded successfully.")
    return embedding_df, embeddings

def save_embeddings_pickle(embeddings: np.ndarray, pickle_path: str):
    logging.info(f"Saving embeddings to pickle file at {pickle_path}...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(embeddings, f)
    logging.info("Embeddings saved successfully.")

def load_embeddings_pickle(pickle_path: str) -> np.ndarray:
    logging.info(f"Loading embeddings from pickle file at {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        embeddings = pickle.load(f)
    logging.info("Embeddings loaded successfully from pickle.")
    return embeddings

def initialize_faiss_flat(embeddings: np.ndarray, dimension: int) -> faiss.IndexFlatL2:
    logging.info("Initializing FAISS IndexFlatL2...")
    index = faiss.IndexFlatL2(dimension)
    logging.info("Adding embeddings to FAISS IndexFlatL2...")
    index.add(embeddings)
    logging.info(f"FAISS IndexFlatL2 initialized with {index.ntotal} vectors.")
    return index

def initialize_faiss_ivf(embeddings: np.ndarray, dimension: int, nlist: int) -> faiss.IndexIVFFlat:
    logging.info("Initializing FAISS IndexIVFFlat...")
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    logging.info("Training FAISS IndexIVFFlat...")
    index.train(embeddings)
    logging.info("Adding embeddings to FAISS IndexIVFFlat...")
    index.add(embeddings)
    logging.info(f"FAISS IndexIVFFlat initialized with {index.ntotal} vectors.")
    return index

def load_or_create_faiss_index(embeddings: np.ndarray, index_path: str, nlist: int = 100) -> faiss.Index:
    num_embeddings = embeddings.shape[0]
    dimension = embeddings.shape[1]

    if CONFIG["CACHE_ENABLED"] and os.path.exists(index_path):
        logging.info("Loading FAISS index from file...")
        index = faiss.read_index(index_path)
        logging.info(f"FAISS index loaded with {index.ntotal} vectors.")
        return index

    # Decide on the FAISS index type based on dataset size
    if num_embeddings < nlist * 10:
        logging.warning(f"Number of embeddings ({num_embeddings}) is less than nlist*10 ({nlist*10}). Adjusting index type.")
        index = initialize_faiss_flat(embeddings, dimension)
    else:
        index = initialize_faiss_ivf(embeddings, dimension, nlist)

    if CONFIG["CACHE_ENABLED"]:
        faiss.write_index(index, index_path)
        logging.info(f"FAISS index saved to {index_path}.")

    return index

def deduplicate_chunks(chunks: List[str]) -> List[str]:
    logging.info("Deduplicating retrieved chunks...")
    unique_chunks = []
    seen = set()
    for chunk in chunks:
        if chunk not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk)
    logging.info(f"Deduplicated to {len(unique_chunks)} unique chunks.")
    return unique_chunks

def summarize_chunks(chunks: List[str], summarizer, max_length: int, min_length: int) -> str:
    logging.info("Summarizing context chunks...")
    combined_text = "\n\n".join(chunks)
    # Handle cases where summarization might fail
    try:
        summary = summarizer(combined_text, max_length=max_length, min_length=min_length, do_sample=False)
        summarized_text = summary[0]['summary_text']
        logging.info("Context summarized successfully.")
        return summarized_text
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        return combined_text  # Fallback to original if summarization fails

def exponential_backoff_retry(func, max_retries: int, backoff_factor: int, *args, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except openai.error.RateLimitError as e:
            wait_time = backoff_factor ** attempt
            logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            break
    return None

# =======================
# Core Functions
# =======================

def get_similar_chunks_by_quarter(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    embedding_df: pd.DataFrame,
    top_k: int = CONFIG["TOP_K_DEFAULT"],
    quarter: Optional[str] = None
) -> List[dict]:
    logging.info(f"Encoding the query: '{query}'")
    query_embedding = model.encode(query, convert_to_tensor=False).astype('float32')
    query_embedding = np.expand_dims(query_embedding, axis=0)

    logging.info("Searching for similar chunks in FAISS index...")
    distances, indices = index.search(query_embedding, top_k * 2)  # Retrieve more for deduplication

    similar_chunks = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(embedding_df):
            continue
        row = embedding_df.iloc[idx]
        if quarter and row['quarter'] != quarter:
            continue
        similar_chunks.append({
            "chunk": row['chunk'],
            "source": row.get('source', 'Unknown'),
            "date": row.get('date', 'Unknown')
        })
        if len(similar_chunks) >= top_k:
            break

    similar_chunks = deduplicate_chunks([chunk['chunk'] for chunk in similar_chunks])
    similar_chunks_with_metadata = []
    for chunk in similar_chunks:
        matched_rows = embedding_df[embedding_df['chunk'] == chunk]
        for _, row in matched_rows.iterrows():
            similar_chunks_with_metadata.append({
                "chunk": chunk,
                "source": row.get('source', 'Unknown'),
                "date": row.get('date', 'Unknown')
            })
            break  # Only take the first match

    logging.info(f"Retrieved {len(similar_chunks_with_metadata)} similar chunks.")
    return similar_chunks_with_metadata

def generate_response(
    prompt: str,
    context_chunks: List[dict],
    summarizer,
    temperature: float = 0.3,
    top_p: float = 0.85
) -> str:
    logging.info("Preparing context for OpenAI...")
    context = "\n\n".join([
        f"Source: {chunk['source']}\nDate: {chunk['date']}\nContent: {chunk['chunk']}"
        for chunk in context_chunks
    ])

    if len(context) > CONFIG["MAX_CONTEXT_LENGTH"]:
        logging.info("Context length exceeds maximum. Summarizing context...")
        context = summarize_chunks(
            [chunk['chunk'] for chunk in context_chunks],
            summarizer,
            CONFIG["SUMMARY_MAX_LENGTH"],
            CONFIG["SUMMARY_MIN_LENGTH"]
        )

    combined_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer (use only information from the context):"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": combined_prompt}
    ]

    def call_openai_api():
        return openai.ChatCompletion.create(
            model=CONFIG["OPENAI_MODEL"],
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=500,
            n=1,
            stop=None
        )

    logging.info("Generating response from OpenAI...")
    response = exponential_backoff_retry(
        call_openai_api,
        CONFIG["MAX_RETRIES"],
        CONFIG["BACKOFF_FACTOR"]
    )

    if response:
        answer = response['choices'][0]['message']['content'].strip()
        logging.info("Response generated successfully.")
        return answer
    else:
        logging.error("Failed to generate a response from OpenAI.")
        return "Sorry, I couldn't generate a response at this time."

# =======================
# Main Function
# =======================

def main():
    logging.info("Starting the Enhanced Study Assistant...")

    # Load or create embeddings
    if CONFIG["CACHE_ENABLED"] and os.path.exists(CONFIG["EMBEDDINGS_PICKLE"]):
        embeddings = load_embeddings_pickle(CONFIG["EMBEDDINGS_PICKLE"])
        embedding_df, _ = load_embeddings(CONFIG["EMBEDDINGS_CSV"])
    else:
        embedding_df, embeddings = load_embeddings(CONFIG["EMBEDDINGS_CSV"])
        if CONFIG["CACHE_ENABLED"]:
            save_embeddings_pickle(embeddings, CONFIG["EMBEDDINGS_PICKLE"])

    num_embeddings, dimension = embeddings.shape
    logging.info(f"Number of embeddings: {num_embeddings}, Dimension: {dimension}")

    # Initialize or load FAISS index
    index = load_or_create_faiss_index(
        embeddings,
        CONFIG["FAISS_INDEX_PATH"],
        CONFIG["FAISS_NLIST"]
    )

    # Initialize SentenceTransformer model
    logging.info(f"Loading SentenceTransformer model: {CONFIG['MODEL_NAME']}...")
    model = SentenceTransformer(CONFIG["MODEL_NAME"])
    logging.info("SentenceTransformer model loaded successfully.")

    # Initialize summarizer
    logging.info(f"Loading summarizer model: {CONFIG['SUMMARIZER_MODEL']}...")
    summarizer = pipeline("summarization", model=CONFIG["SUMMARIZER_MODEL"])
    logging.info("Summarizer loaded successfully.")

    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logging.critical("OpenAI API Key is not set. Please set the OPENAI_API_KEY environment variable.")
        return

    logging.info("Enhanced Study Assistant is ready to receive your questions.")

    while True:
        try:
            user_prompt = input("\nPlease ask your question (type 'exit' to quit): ").strip()
            if user_prompt.lower() == 'exit':
                logging.info("Goodbye!")
                break

            # Dynamic quarter selection
            user_quarter = input("Enter the quarter (e.g., 2022Q4) or press Enter to include all: ").strip()
            quarter = user_quarter if user_quarter else None

            # Adjust top_k based on query length (example logic)
            top_k = CONFIG["TOP_K_DEFAULT"] + min(len(user_prompt.split()), CONFIG["TOP_K_MAX"])

            similar_chunks = get_similar_chunks_by_quarter(
                user_prompt,
                model,
                index,
                embedding_df,
                top_k=top_k,
                quarter=quarter
            )

            if not similar_chunks:
                print("No relevant information found for your query.")
                continue

            answer = generate_response(user_prompt, similar_chunks, summarizer)
            print(f"\nAnswer: {answer}")

        except KeyboardInterrupt:
            logging.info("Interrupted by user. Exiting...")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            print("An error occurred. Please try again.")

# =======================
# Entry Point
# =======================

if __name__ == "__main__":
    main()
