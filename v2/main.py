import streamlit as st
import pandas as pd

st.title("Hello from OverflowQuest!")
st.write("Search your stackoverflow questions here!")


@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    tf_idf_matrix = pd.read_csv('tf-idf-matrix.csv', index_col=0)
    return train_df, tf_idf_matrix


train_df, tf_idf_matrix = load_data()

query = st.text_input("Enter your query:")

if query:
    query_words = query.lower().split()

    matched_docs = pd.DataFrame()

    for word in query_words:
        if word in tf_idf_matrix.index:
            matched_docs = pd.concat(
                [matched_docs, tf_idf_matrix.loc[word]], axis=1)

    if not matched_docs.empty:
        doc_scores = matched_docs.sum(axis=1)
        doc_scores = doc_scores[doc_scores > 0].sort_values(ascending=False)
        doc_ids = doc_scores.index.astype(int)

        st.write(f"Documents matching '{query}':")

        for doc_id in doc_ids[:11]:
            doc = train_df.iloc[doc_id]            
            colored_score_text = f'<span style="color: green; font-weight: bold;"><b>Score:</b> {doc_scores[str(doc_id)]}</span>'




            st.markdown(colored_score_text, unsafe_allow_html=True)
            st.write(f"**ID:** {doc['Id']}")
            st.write(f"**Created At:** {doc['CreationDate']}")
            st.write(f"**Title:** {doc['Title']}")
            st.write(f"**Body:** {doc['Body']}")
            st.write("---")
    else:
        st.write("No matching documents found.")