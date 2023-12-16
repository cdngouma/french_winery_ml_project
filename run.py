import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.cluster import AgglomerativeClustering

def preprocess_text(text):
    # Remove numerical tokens
    text = re.sub(r'\b\d+\b', '', text)
    return text

def custom_tokenizer(text):
    # Custom tokenizer to preprocess text and extract tokens
    text = preprocess_text(text)
    return text.split()

def encode_texts(data):
    # Custom stop words
    descriptions = data.apply(lambda x: re.sub(f"{re.escape(x['variety'])}|{x['winery']}", "", x["description"]), axis=1)
    # Apply TF-IDF vectorization with binary encoding
    tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, stop_words=text.ENGLISH_STOP_WORDS, binary=True)
    encoded_documents = tfidf_vectorizer.fit_transform(descriptions)
    return encoded_documents

winemag_clean = pd.read_csv("winemag_clean.csv")

encoded_docs = encode_texts(winemag_clean)

# Perform hierarchical clustering
cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, linkage='ward') # Adjust distance_threshold as needed
clusters = cluster.fit_predict(encoded_docs.toarray())

data["cluster"] = clusters

data.to_csv('clusters.csv', index=False)