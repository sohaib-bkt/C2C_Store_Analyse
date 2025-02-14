import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import os
from slider import add_sidebar_links 

st.title("🧹 Nettoyage des Données")
add_sidebar_links()

file_path = "data/user_data.csv"
if os.path.exists(file_path):
    db = pd.read_csv(file_path)
    
    # 🔎 Affichage des valeurs manquantes
    st.write("### Valeurs manquantes")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(db.isna(), cbar=False, ax=ax)
    st.pyplot(fig)


    # Sélection des colonnes numériques
    colonnes_numeriques = db.select_dtypes(include='number').columns

    # Configuration de la figure
    fig = plt.figure(figsize=(25, 6))

    # Création des boxplots
    db[colonnes_numeriques].plot(
        kind='box',
        subplots=True,
        layout=(1, len(colonnes_numeriques)),
        ax=fig.gca()  # Utilise l'axe de la figure créée
    )

    # Ajustement de l'espacement
    plt.tight_layout()

    # Affichage dans Streamlit
    st.pyplot(fig)

    # 🚀 Suppression des doublons
    db.drop_duplicates(subset='identifierHash', keep='first', inplace=True)

    # 📌 Suppression des colonnes inutiles
    cols_to_drop = ['identifierHash', 'type', 'country', 'gender', 'civilityTitle']
    db1 = db.drop(cols_to_drop, axis=1)

    # 📌 Traitement des valeurs aberrantes
    colonnes_numeriques = ['daysSinceLastLogin', 'seniority', 'seniorityAsMonths', 'seniorityAsYears']
    for col in colonnes_numeriques:
        mean = db1[col].mean()
        std = db1[col].std()
        z_score = (db1[col] - mean) / std
        db1 = db1[np.abs(z_score) <= 3]

    # 📌 Encodage des variables catégoriques
    string_columns = ['language', 'countryCode', 'hasAnyApp', 'hasAndroidApp', 'hasIosApp', 'hasProfilePicture']
    ordinal_encoder = OrdinalEncoder()
    encoded_data = ordinal_encoder.fit_transform(db[string_columns])
    db_encoded = pd.DataFrame(encoded_data, columns=[col + '_encoded' for col in string_columns])
    db2 = db1.join(db_encoded).drop(string_columns, axis=1)

    # ✅ Sauvegarde des versions nettoyées
    db1.to_csv("data/user_data_cleaned.csv", index=False)
    db2.to_csv("data/user_data_encoded.csv", index=False)

    st.success("✅ Nettoyage terminé et fichiers sauvegardés !")

    st.write("### Aperçu des données nettoyées")
    st.dataframe(db1.head())

    st.write("### Aperçu des données encodées")
    st.dataframe(db2.head())
else:
    st.error(f"⚠️ Fichier introuvable : {file_path}")
