import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from slider import add_sidebar_links

add_sidebar_links()

# 🏆 Titre de l'application
st.title("🔬 Clustering et Dendrogramme des Utilisateurs")

# 📂 Chargement des données
file_path = "data/user_data.csv"  # Chemin vers le fichier
if not file_path:
    st.error("❌ Fichier introuvable. Vérifiez le chemin.")
else:
    db = pd.read_csv(file_path)

    # 🏗️ Prétraitement des données
    st.write("### 📊 Prétraitement des données")
    st.write(f"📌 **Taille du dataset d'origine**: {db.shape}")

    # Supposons que `db3` soit un dataset filtré des variables inutiles
    db3 = db.select_dtypes(include='number')  # Suppression des colonnes non numériques
    db_final = db3.sample(frac=0.01)  # Sélection aléatoire de 1 % des données

    st.write(f"📉 **Taille après filtrage**: {db3.shape}")
    st.write(f"📦 **Données finales après réduction**: {db_final.shape}")

    # 📊 Affichage des colonnes sélectionnées
    st.write("🔍 **Colonnes conservées**:")
    st.write(db_final.columns.tolist())

    # 🔢 Vérification des valeurs numériques avant clustering
    if db_final.empty:
        st.error("⚠️ Les données sélectionnées sont vides après filtrage.")
    else:
        # 📈 Affichage du dendrogramme
        st.write("### 🌿 Dendrogramme")
        fig, ax = plt.subplots(figsize=(11, 8))
        sch.dendrogram(sch.linkage(db_final, method='ward'), ax=ax)
        plt.title("Dendrogramme des utilisateurs")
        plt.xlabel("Utilisateurs")
        plt.ylabel("Distance")
        st.pyplot(fig)

        # 💡 Affichage du score de silhouette
        st.write("### 📏 Score de Silhouette"
                 "\nLe score de silhouette mesure la qualité des clusters obtenus.")
        scores = []
        for n_clusters in range(2, 11):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            preds = clusterer.fit_predict(db_final)
            score = silhouette_score(db_final, preds)
            scores.append(score)

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(range(2, 11), scores, marker='o')
        plt.title("Score de Silhouette pour le clustering")
        plt.xlabel("Nombre de clusters")
        plt.ylabel("Score de Silhouette")
        st.pyplot(fig)


        # 💡 Recommandation de clustering
        st.info(
            "📌 Selon le dendrogramme, **le nombre recommandé de clusters serait de deux** (Actif/Inactif). "
            "Cependant, nous pourrions explorer plus de clusters pour affiner la segmentation. 🚀"
        )
