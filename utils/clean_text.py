import pandas as pd
import numpy as np
import plotly.express as px
pd.options.plotting.backend = 'plotly'
from itertools import chain
import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import openai
import os
from pathlib import Path
import sys

main_path = Path(__file__).resolve().parent.parent.parent.parent
if str(main_path) not in sys.path:
    sys.path.append(str(main_path))

def split_text_nltk(text, max_sentences=50):
    """
    Splits text into chunks based on a maximum number of sentences.
    """
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = ' '.join(sentences[i:i + max_sentences])
        chunks.append(chunk)
    return chunks

def get_similar_chunks(query, index, embedding_df, top_k=5):
    '''Function to get top N similar chunks'''
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_tensor=False).astype('float32')
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, top_k)
    similar_chunks = embedding_df.iloc[indices[0]]['chunk'].tolist()
    return similar_chunks

def generate_response(prompt, context_chunks, api_key):
    '''Function to generate a response using OpenAI's GPT'''
    
    # Combine the context chunks into a single string
    context = "\n\n".join(context_chunks)
    openai.api_key = api_key
    
    # Define the prompt for the language model
    combined_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
    
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=combined_prompt,
        max_tokens=150,
        temperature=0.7,
        n=1,
        stop=None,
    )
    
    answer = response.choices[0].text.strip()
    return answer

def transform_text(text):
    '''First step neccesssary conversions'''

    format = '%m/%d/%y'
    text['Time'] = pd.to_datetime(text['Time'],format=format)
    
    text['Quarter'] = text['Time'].dt.to_period('Q')
    text['Month'] = text['Time'].dt.month_name()


    # grouped_text = text.groupby('Time').sum().reset_index()[['Time', 'Study Materials']]
    
    text = separate_week_from_notes(text)

   # Group by 'Time' and aggregate textual columns by concatenating their values
    grouped_text = text.groupby('Time', as_index=False).agg({
        'Quarter': "first",
        'Month': "first",
        'Notes': lambda x: ' | '.join(x.dropna().astype(str)),
        'Study Materials': lambda x: ' | '.join(x.dropna().astype(str)),
        'Study Materials (Lecture Counted)': lambda x: ' | '.join(x.dropna().astype(str))
    })
    
    # Combine the two study materials columns into a single column
    # Using a separator ' | ' to combine both columns. Adjust as needed.
    grouped_text['Study Materials Combined'] = grouped_text.apply(
        lambda row: ' | '.join(filter(None, [row['Study Materials'], row['Study Materials (Lecture Counted)']])), 
        axis=1
    )
    
    # Drop the old separate study materials columns
    grouped_text = grouped_text.drop(columns=['Study Materials', 'Study Materials (Lecture Counted)'])
    
    # Rename 'Study Materials Combined' back to 'Study Materials' if desired
    grouped_text = grouped_text.rename(columns={'Study Materials Combined': 'Study Materials'})
    
    grouped_text['Study Materials'] = grouped_text['Time'].astype(str) + ' | ' + grouped_text['Quarter'].astype(str) + ' | ' + grouped_text['Month'].astype(str) + ' | ' + grouped_text['Study Materials']
    
    grouped_text.drop(columns=['Notes'],inplace=True)
    
    return grouped_text


def separate_week_from_notes(df):
    def process_row(row):
        note = row['Notes']
        study_mat = row['Study Materials']

        # If Notes is empty or NaN, do nothing
        if pd.isna(note) or note.strip() == '':
            return row

        # Regex to check if the note starts with "Week" followed by a number
        # and optionally more text after that.
        match = re.match(r'^(Week\s+\d+)(.*)$', note.strip(), flags=re.IGNORECASE)
        if match:
            # Extract "Week ..." part and the remainder
            week_part = match.group(1).strip()  # e.g. "Week 8"
            remainder = match.group(2).strip()   # everything after "Week 8"

            # Keep "Week ..." in Notes
            row['Notes'] = week_part

            # If there's more text after the week info, move it to Study Materials
            if remainder:
                if pd.isna(study_mat) or study_mat.strip() == '':
                    row['Study Materials'] = remainder
                else:
                    row['Study Materials'] = study_mat + ' | ' + remainder
        else:
            # No week info found, move entire Notes content to Study Materials
            if pd.isna(study_mat) or study_mat.strip() == '':
                row['Study Materials'] = note
            else:
                row['Study Materials'] = study_mat + ' | ' + note

            # Clear the Notes column since it didn't contain a Week reference
            row['Notes'] = ''

        return row

    # Apply the transformation row by row
    df = df.apply(process_row, axis=1)

    return df


def tokenize(grouped_text):
    '''Tokenize the text data'''
    split_text = (grouped_text['Study Materials']
                  .str.lower()
                  .str.replace(r'\([\d]*m\)','',regex=True)
                  .str.replace(',','')
                  .str.replace(' ','_')
                  .str.strip()
                  .str.split(r'_(?![\d]+)')) # keep code

    tokens = split_text.explode().to_list()
    return tokens