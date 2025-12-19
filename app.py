import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI ---
MAX_SEQ_LENGTH = 256
TOP_K = 10

# --- FUNGSI CUSTOM CSS (DARK/LIGHT MODE) ---
def inject_theme(is_dark_mode):
    if is_dark_mode:
        # Dark Mode CSS
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #0e1117;
                color: #fafafa;
            }
            .stTextInput > div > div > input {
                color: #fafafa;
                background-color: #262730;
            }
            div[data-testid="stExpander"] {
                background-color: #262730;
                border: 1px solid #4f4f4f;
                color: #fafafa;
            }
            p, h1, h2, h3, div {
                color: #fafafa;
            }
            a {
                color: #4da6ff !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        # Light Mode CSS
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #ffffff;
                color: #000000;
            }
            div[data-testid="stExpander"] {
                background-color: #f0f2f6;
                border: 1px solid #d6d6d6;
                color: #000000;
            }
            p, h1, h2, h3, div {
                color: #31333F;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# --- FUNGSI LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    print("Memuat model dan data...")
    
    # 1. Load Tokenizer (Original Logic)
    with open('models/tokenizer.json', 'r') as f:
        file_content = f.read()
        try:
            parsed = json.loads(file_content)
        except json.JSONDecodeError:
            parsed = file_content

        if isinstance(parsed, str):
            tokenizer = tokenizer_from_json(parsed)
        elif isinstance(parsed, dict):
            json_string = json.dumps(parsed)
            tokenizer = tokenizer_from_json(json_string)
        else:
            tokenizer = tokenizer_from_json(file_content)
        
    # 2. Load Model
    model = tf.keras.models.load_model('models/encoder_model_best_grid.keras', compile=False)
    
    # 3. Load Vektor Indeks (English Only)
    index_vectors = np.load('models/index_vectors_vanilla.npy')
    
    # 4. Load Metadata (English Only)
    df = pd.read_csv('models/korpus_indeks_10YR_15k.csv', engine='python')
    
    return tokenizer, model, index_vectors, df

# --- FUNGSI PREPROCESSING ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- SEARCH ENGINE ---
def search_engine(query, tokenizer, model, index_vectors, df):
    # 1. Preprocess Query
    cleaned_query = clean_text(query)
    seq = tokenizer.texts_to_sequences([cleaned_query])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    
    # 2. Predict Vector
    query_vector = model.predict(padded, verbose=0)
    query_vector = query_vector / (np.linalg.norm(query_vector, axis=1, keepdims=True) + 1e-12)
    
    # 3. Cosine Similarity
    sims = cosine_similarity(query_vector, index_vectors)[0]
    
    # 4. Sort Top K
    top_indices = sims.argsort()[::-1][:TOP_K]
    
    results = []
    for idx in top_indices:
        results.append({
            "score": float(sims[idx]),
            "title": df.iloc[idx]['Judul'],
            "url": df.iloc[idx]['URL'],
            "abstract": df.iloc[idx]['Abstrak']
        })
    return results

# --- UI STREAMLIT ---
st.set_page_config(page_title="Bilingual Search", layout="wide")

# Sidebar for Theme Settings
with st.sidebar:
    st.header("⚙️ Settings")
    is_dark = st.toggle("🌙 Dark Mode", value=True)
    inject_theme(is_dark)

st.title("🇮🇩 🇺🇸 Cross-Lingual Semantic Search")
st.markdown("Cari jurnal dalam **Bahasa Indonesia** atau **Inggris**.")

# Load data
tokenizer, model, index_vectors, df = load_artifacts()

# Input User
query = st.text_input("Masukkan kata kunci pencarian...", placeholder="Contoh: Deteksi anomali pada jaringan IoT")

if query:
    with st.spinner('Mencari paper yang relevan...'):
        results = search_engine(query, tokenizer, model, index_vectors, df)
    
    st.write(f"Menemukan hasil untuk: **{query}**")
    
    for i, res in enumerate(results):
        with st.expander(f"{i+1}. {res['title']} (Score: {res['score']:.4f})"):
            st.markdown(f"**URL:** [Link Paper]({res['url']})")
            st.write(f"**Abstrak:** {str(res['abstract'])[:300]}...")