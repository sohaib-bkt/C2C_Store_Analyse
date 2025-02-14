import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from slider import add_sidebar_links

add_sidebar_links()

# 🎯 Titre de l'application
st.title("🤖 Segmentation des Vendeurs avec K-Means")

# 📂 Chargement des données
file_path = "data/seller.csv"  # Adapter si nécessaire
if not file_path:
    st.error("❌ Fichier introuvable. Vérifiez le chemin.")
else:
    sellers = pd.read_csv(file_path)

    # 🎯 Sélection des colonnes pertinentes
    cols = ['meanproductssold', 'meanproductslisted', 'meansellerpassrate']
    st.write("### 🔍 Colonnes sélectionnées")
    st.write(cols)
    
    db_final_subset = sellers[cols].dropna()

    st.write("### 🗃️ Aperçu des données")
    st.dataframe(db_final_subset.head())

    # 📏 Standardisation des données (Z-score)
    z_scores = (db_final_subset - db_final_subset.mean()) / db_final_subset.std()

    # 🚨 Filtrage des valeurs aberrantes (Z-score > 3)
    db_final_subset = db_final_subset[(z_scores < 3).all(axis=1)]

    # 🔎 Détection des valeurs aberrantes via l'IQR
    Q1 = db_final_subset.quantile(0.25)
    Q3 = db_final_subset.quantile(0.75)
    IQR = Q3 - Q1
    db_final_subset = db_final_subset[~((db_final_subset < (Q1 - 1.5 * IQR)) | (db_final_subset > (Q3 + 1.5 * IQR))).any(axis=1)]

    # 🔄 Normalisation des données
    scaler = StandardScaler()
    db_final_scaled = scaler.fit_transform(db_final_subset)

    # 📊 Choix optimal du nombre de clusters (k) avec silhouette score
    k_values = range(2, 11)  # Tester k entre 2 et 10
    silhouettes = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(db_final_scaled)
        silhouettes.append(silhouette_score(db_final_scaled, kmeans.labels_))

    # 🎯 Trouver le k optimal
    optimal_k = k_values[silhouettes.index(max(silhouettes))]
    st.success(f"💡 Nombre optimal de clusters basé sur le score de silhouette : **{optimal_k}**. Ainsi, on peut classer les vendeurs en 3 catégories : des vendeurs performants, des vendeurs intermédiaires et des vendeurs peu actifs.")

    # 📈 Visualisation du score silhouette
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_values, silhouettes, marker="o", linestyle="-", color="blue")
    ax.set_xlabel("Nombre de clusters (k)")
    ax.set_ylabel("Score de silhouette")
    ax.set_title("📉 Score de silhouette en fonction de k")
    st.pyplot(fig)

    # 🔧 Ajout d'un slider pour choisir manuellement k
    st.write("### 🔧 Ajustez le nombre de clusters (k)")
    k_manual = st.slider("Sélectionnez la valeur de k", min_value=2, max_value=10, value=optimal_k, step=1)

    # 🚀 Application de K-Means avec le k sélectionné
    kmeans = KMeans(n_clusters=k_manual, random_state=42, n_init=10)
    kmeans.fit(db_final_scaled)
    labels = kmeans.labels_

    # 🎨 Réduction de dimension avec PCA pour visualisation
    pca = PCA(n_components=2)
    db_final_pca = pca.fit_transform(db_final_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(db_final_pca[:, 0], db_final_pca[:, 1], c=labels, cmap='viridis', alpha=0.8)
    ax.set_title(f"📌 Clustering des Vendeurs avec k = {k_manual}")
    ax.set_xlabel("Composante principale 1")
    ax.set_ylabel("Composante principale 2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)

    # 📝 Interprétation des résultats
    st.write("### 📌 Interprétation des Résultats")
    st.markdown(f"""
    - 🚀 **Nombre de clusters utilisé :** {k_manual}
    - 🔹 **Classification des vendeurs en {k_manual} groupes distincts**  
      Les caractéristiques de chaque groupe pourront être approfondies en fonction des indicateurs sélectionnés :
      - **meanproductssold** : Nombre moyen de produits vendus
      - **meanproductslisted** : Nombre moyen de produits listés
      - **meansellerpassrate** : Taux de réussite du vendeur
    - 🔬 **Méthodologie :**  
      - **Sélection des colonnes :** {', '.join(cols)}  
      - **Standardisation et filtrage des données** (Z-score & IQR)  
      - **Utilisation de l'algorithme K-Means**  
      - **Détermination du k optimal** via le score silhouette  
      - **Ajustement manuel possible** grâce au slider  
      - **Visualisation** via une réduction en 2D avec PCA  
    """)
