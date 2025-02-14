import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from slider import add_sidebar_links

add_sidebar_links()

# Charger les données (remplace db3 et db2 par tes propres DataFrames)
db3 = pd.read_csv("data/user_data.csv")
db2 = pd.read_csv("data/user_data.csv")

st.title("⚙️ Transformation des Données")
##  **Standardisation des Données**
st.subheader("📏 Standardisation des Données")

# Sélectionner uniquement les colonnes numériques
X = db2.select_dtypes(include='number')

# Standardiser les données
scaler = StandardScaler()
db2_scaled = scaler.fit_transform(X)
db2_scaled = pd.DataFrame(db2_scaled, columns=X.columns)

st.write("**Données après standardisation :**")
st.dataframe(db2_scaled.head())  # Afficher un aperçu des données standardisées

##  **Ingénierie des fonctionnalités (Détection des Bots)**
st.subheader("🔍 Détection des Bots")

# Définir les critères des bots
potential_bots = db3[
    (db3['socialNbFollowers'] > db3['socialNbFollowers'].quantile(0.99)) & 
    (db3['productsBought'] < 1) & 
    (db3['daysSinceLastLogin'] > 90)
]
db3['isBot'] = db3.index.isin(potential_bots.index)

# Afficher les résultats sous forme de texte
st.write("nous avons détecté les bots et stocké leur statut dans la colonne 'isBot'")
st.write("**Critères de détection des bots :**")
st.write("- Nombre de followers > 99e percentile")
st.write("- Aucun produit acheté")
st.write("- Plus de 90 jours depuis la dernière connexion")
st.write("**Nombre d'utilisateurs détectés comme bots :**", db3['isBot'].sum())
st.write("**Nombre d'utilisateurs légitimes :**", (~db3['isBot']).sum())

# Afficher un diagramme circulaire
fig, ax = plt.subplots()
db3['isBot'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, labels=["Légitimes", "Bots"], colors=["lightblue", "red"])
ax.set_ylabel('')
st.pyplot(fig)

st.write("> La majorité des utilisateurs sont considérés comme légitimes, tandis qu'une très petite fraction est détectée comme des bots selon les critères définis.")


