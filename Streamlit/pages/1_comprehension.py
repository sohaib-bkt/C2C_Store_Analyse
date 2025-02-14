import streamlit as st
import pandas as pd
import os
from slider import add_sidebar_links 

st.title("📊 Compréhension des Données")
add_sidebar_links()
file_path = "data/user_data.csv"
if os.path.exists(file_path):
    db = pd.read_csv(file_path)
    st.write("✅ Données brutes chargées :", file_path)
    st.dataframe(db.head())

    st.write("### Distribution des types de données")
    st.bar_chart(db.dtypes.value_counts())

else:
    st.error(f"⚠️ Fichier introuvable : {file_path}")
