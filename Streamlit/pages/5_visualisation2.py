import pandas as pd
import os
import streamlit as st
import matplotlib.pyplot as plt
from slider import add_sidebar_links 

add_sidebar_links()

# Charger les données (remplace par ton propre dataset)
db3 = pd.read_csv("data/user_data_encoded.csv")

st.title("📊 Analyse des Acheteurs et Vendeurs")

### 1️⃣ Informations sur les Acheteurs
st.header("🔍 Informations sur les Acheteurs")

buyers_db = db3[db3['productsBought'] > 0]
st.write(f"**Nombre total d'acheteurs :** {buyers_db.shape[0]}")
st.write(f"**Nombre moyen de produits achetés :** {buyers_db.productsBought.mean():.2f}")

# Acheteurs réussis
Sbuyers_db = db3[db3.productsBought >= 3]
st.write(f"**Acheteurs réussis (≥3 achats) :** {Sbuyers_db.shape[0]}")

# Affichage des statistiques
st.subheader("📈 Statistiques des achats")
st.write("**Produits achetés :**", buyers_db.productsBought.describe())
st.write("**Nombre de followers des acheteurs réussis :**", Sbuyers_db.socialNbFollowers.describe())

### 2️⃣ Informations sur les Vendeurs
st.header("🛒 Informations sur les Vendeurs")

successful_sellers_db = db3[db3['productsSold'] > 0]
st.write(f"**Nombre de vendeurs actifs :** {successful_sellers_db.shape[0]}")
st.write(f"**Nombre moyen de produits vendus :** {successful_sellers_db.productsSold.mean():.2f}")

# Vendeurs à succès
Ssellers_db = db3[db3.productsSold >= 6]
st.write(f"**Vendeurs à succès (≥6 ventes) :** {Ssellers_db.shape[0]}")

# Affichage des statistiques
st.subheader("📈 Statistiques des ventes")
st.write("**Produits vendus :**", successful_sellers_db.productsSold.describe())

### 3️⃣ Informations sur les Produits
st.header("📦 Analyse de la Qualité des Produits")
st.write("La qualité des produits est un facteur clé pour les vendeurs et les acheteurs. ")
st.write("Nous allons analyser la qualité des produits en fonction de leur taux de réussite.")

# Catégorisation des vendeurs par qualité
productsH_db = db3[db3.productsPassRate >= 90]
productsMh_db = db3[(db3.productsPassRate >= 80) & (db3.productsPassRate < 90)]
productsS_db = db3[(db3.productsPassRate >= 60) & (db3.productsPassRate < 80)]
productsU_db = db3[db3.productsPassRate < 60]

st.write(f"**Vendeurs de haute qualité (≥90% PassRate) :** {productsH_db.shape[0]}")
st.write(f"**Vendeurs de qualité moyenne-haute (80-90%) :** {productsMh_db.shape[0]}")
st.write(f"**Vendeurs de qualité standard (60-80%) :** {productsS_db.shape[0]}")
st.write(f"**Vendeurs de faible qualité (<60%) :** {productsU_db.shape[0]}")


# Graphique des ventes par qualité
fig, ax = plt.subplots()
labels = ['Haute Qualité', 'Moyenne-Haute', 'Standard', 'Faible Qualité']
values = [productsH_db.shape[0], productsMh_db.shape[0], productsS_db.shape[0], productsU_db.shape[0]]
ax.bar(labels, values, color=['green', 'blue', 'orange', 'red'])
ax.set_ylabel("Nombre de Vendeurs")
ax.set_title("Répartition des vendeurs par qualité des produits")
st.pyplot(fig)


# ============================
# 3️⃣ Répartition de la Qualité des Vendeurs
# ============================
st.header("📊 Répartition de la Qualité des Vendeurs")
st.write("Nous allons maintenant analyser la qualité des vendeurs en fonction de la qualité de leurs produits.")

# Catégorisation des vendeurs en fonction de la qualité des produits
productsH_db = db3[db3.productsPassRate >= 90]  # Haute qualité
productsMh_db = db3[(db3.productsPassRate >= 80) & (db3.productsPassRate < 90)]  # Qualité moyenne-haute
productsS_db = db3[(db3.productsPassRate >= 60) & (db3.productsPassRate < 80)]  # Qualité standard
productsU_db = db3[(db3.productsPassRate < 60) & (db3.productsPassRate > 0)]  # Faible qualité

# Comptage des vendeurs dans chaque catégorie
quality_counts = {
    "Haute qualité (≥90%)": productsH_db.shape[0],
    "Moyenne-haute (80-90%)": productsMh_db.shape[0],
    "Standard (60-80%)": productsS_db.shape[0],
    "Faible qualité (<60%)": productsU_db.shape[0]
}

# Création du graphique
fig, ax = plt.subplots()
ax.pie(quality_counts.values(), labels=quality_counts.keys(), autopct='%1.1f%%', colors=["#4CAF50", "#FFC107", "#2196F3", "#F44336"])
ax.set_title("Répartition des vendeurs par qualité des produits")

# Affichage du graphique dans Streamlit
st.pyplot(fig)

st.subheader("📌 Moyenne des ventes par catégorie de qualité")
sellers_db = db3[(db3.productsListed > 0) | (db3.productsSold > 0)]

st.write("**Les vendeurs de haute qualité vendent en moyenne plus de produits (57.7%) que les autres catégories, tandis que les vendeurs de faible qualité en vendent très peu (9.9%).**")



