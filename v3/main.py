import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity


st.title("Hello from OverflowQuest!")
st.write("Search your Stack Overflow questions here using LSI!")


@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    topic_encoded_df = pd.read_csv('lsa-topic-encoded.csv')
    encoding_matrix = pd.read_csv('lsa-encoding-matrix.csv', index_col=0)
    return train_df, topic_encoded_df, encoding_matrix

train_df, topic_encoded_df, encoding_matrix = load_data()


query = st.text_input("Enter your query:")

if query:
    def preprocess_query(query, dictionary):
        clean_query = re.sub(r"[^a-z\s']", " ", query.lower())
        query_vector = np.zeros(len(dictionary))
        for word in clean_query.split():
            if word in dictionary:
                query_vector[dictionary.index(word)] += 1
        return query_vector

    dictionary = list(encoding_matrix.index)
    query_vector = preprocess_query(query, dictionary)

    query_lsa = np.dot(query_vector, encoding_matrix.values)

    document_topics = topic_encoded_df.iloc[:, :-1].values
    similarities = cosine_similarity([query_lsa], document_topics)[0]

    top_indices = similarities.argsort()[::-1][:10]
    top_scores = similarities[top_indices]

    st.write(f"Documents matching '{query}':")

    if top_scores.any():
        for idx, score, i in zip(top_indices, top_scores, range(len(top_indices))):
            doc = train_df.iloc[idx]
            st.write(int(i))
            st.write(int(idx))
            st.markdown(f"<span style='color: green; font-weight: bold;'>Score: {score:.4f}</span>", unsafe_allow_html=True)
            st.write(f"**ID:** {doc['Id']}")
            st.write(f"**Created At:** {doc['CreationDate']}")
            st.write(f"**Title:** {doc['Title']}")
            st.write(f"**Body:** {doc['Body']}")
            st.write("---")
    else:
        st.write("No matching documents found.")
