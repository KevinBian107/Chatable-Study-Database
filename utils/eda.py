import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
pd.options.plotting.backend = 'plotly'
from itertools import chain
import re

def transform_study(study):
    '''First step neccesssary conversions'''

    # Timestamp Conversion
    format = '%m/%d/%y'
    study['date'] = pd.to_datetime(study['date'],format=format)

    # Group by date to get unique data set on dates
    grouped_study = study.groupby('date').mean().fillna(0).reset_index()


    # add columns
    grouped_study = grouped_study.assign(math18 = grouped_study['math18hw']
                                         +grouped_study['math18review']
                                         +grouped_study['math18matlab'])
    grouped_study = grouped_study.assign(dsc10 = grouped_study['dsc10review']
                                         +grouped_study['dsc10hw'])
    grouped_study = grouped_study.assign(math20b = grouped_study['math20breview']
                                         +grouped_study['math20bhw'])

    # drop and rename
    grouped_study = (grouped_study
                     .drop(columns=['math18hw','math18review','math18matlab','dsc10review','dsc10hw','math20breview','math20bhw'])
                     .rename(columns={'ds':'ds_summer_project'})
                     )
    
    # convert
    number_col = grouped_study.select_dtypes(include='number').columns
    grouped_study[number_col] = grouped_study[number_col] / 60

    grouped_study['week'] = grouped_study['week'] * 60

    return grouped_study



def transform_text(text):
    '''First step neccesssary conversions'''

    format = '%m/%d/%y'
    text['Time'] = pd.to_datetime(text['Time'],format=format)

    grouped_text = text.groupby('Time').sum().reset_index()[['Time', 'Study Materials']]
    return grouped_text



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