# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:09:18 2023

@author: Prakhar Agrawal
"""

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():
    return pd.read_csv("ayurvedic-colleges_2- (1).csv")

def main():
    
    st.set_page_config(
        page_title="Ayurvedic College Recommender",
        page_icon="🎓",
        layout="wide",
    )

    
    st.title("Ayurvedic College Recommender")
    st.write("Enter your state, and we'll recommend Ayurvedic colleges for you.")

    
    dataset = load_data()
    dataset = dataset.drop_duplicates(subset='College ID')

    
    user_input = st.text_input("Enter your state name:")

    if not user_input:
        st.warning("Please enter a state name.")
        st.stop()

    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset['State'])
    user_matrix = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_matrix, tfidf_matrix)
    similar_indices = [i for i, score in enumerate(similarity[0]) if score > 0.5]

    
    if similar_indices:
        st.subheader("Recommended Colleges:")
        for idx in similar_indices:
            st.markdown(f"<span style='color: #A633FF;'>College ID:</span> {dataset.iloc[idx]['College ID']}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #FFB533;'>Name of the College:</span> {dataset.iloc[idx]['Name of the College']}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #33FF45;'>State:</span> {dataset.iloc[idx]['State']}", unsafe_allow_html=True)
            st.markdown("---", unsafe_allow_html=True)  
    else:
        st.warning("No records found for the given state.")

if __name__ == "__main__":
    main()
