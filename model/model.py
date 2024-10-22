import pickle
import re
from pathlib import Path
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

classes = ['Arabic', 'Danish', 'Dutch' ,'English', 'French', 'German','Greek', 'Hindi'
 'Italian', 'Kannada' ,'Malayalam', 'Portugeese' ,'Russian', 'Spanish',
 'Sweedish' ,'Tamil', 'Turkish']

__version__ = "0.1.0"

Base_Dir = Path(__file__).resolve(strict=True).parent

with open (f"model/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)



def predict_pipeline(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'\[|\]', ' ', text)
    text = text.lower()
    pred = model.predict([text])
    return classes[pred[0]]