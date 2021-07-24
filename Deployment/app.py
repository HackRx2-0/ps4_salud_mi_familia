import streamlit as st
import pickle as pk
from Embeddings import *
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

with open('Embedd.pk', 'rb') as f:
    embedding = pk.load(f)
query = pd.read_csv('query_df.csv')['0'].to_list()
st.cache()
st.title('Bajaj Finserve Recommendation Engine')
st.header('Salud Mi Familia')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

# icon("search")
selected = st.text_input("")
button_clicked = st.button("OK")

query_embedings = SentenceEmbedding(selected)
query_embedings.get_embeddings()

scores = []
for text in embedding:
    scores.append(r2_score(query_embedings.embedding, text))

ranks = np.array(scores).argsort()[-10:][::-1]
for i in ranks[:2]:
    st.write(query[i])