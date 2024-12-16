import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
pd.options.plotting.backend = 'plotly'
from itertools import chain
import re

def transform_text(text):
    '''First step neccesssary conversions'''

    format = '%m/%d/%y'
    text['Time'] = pd.to_datetime(text['Time'],format=format)

    # grouped_text = text.groupby('Time').sum().reset_index()[['Time', 'Study Materials']]
    
    text = separate_week_from_notes(text)

   # Group by 'Time' and aggregate textual columns by concatenating their values
    grouped_text = text.groupby('Time', as_index=False).agg({
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