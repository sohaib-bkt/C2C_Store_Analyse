import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from slider import add_sidebar_links

add_sidebar_links()

# 🏆 Titre de l'application
st.title("🔬 Analyse des Corrélations entre Variables")

# 📂 Chargement des données
file_path = "data/seller.csv"  # Chemin vers le fichier
if not file_path:
    st.error("❌ Fichier introuvable. Vérifiez le chemin.")
else:
    db = pd.read_csv(file_path)

    # 🏗️ Prétraitement des données
    st.write("### 📊 Prétraitement des Données")
    st.markdown("""  
    **Nous passons à la table concernant les vendeurs existants sur ce site.   
    Ce jeu de données a été extrait de la même source (plateforme C2C).**  """)  
    db_numeric = db.select_dtypes(include='number')  # Sélection des colonnes numériques

    if db_numeric.empty:
        st.error("⚠️ Aucune donnée numérique trouvée pour l'analyse.")
    else:
        # 🔢 Calcul de la matrice de corrélation
        correlation_matrix = db_numeric.corr()

        # 📈 Affichage de la Heatmap des Corrélations
        st.write("### 🔥 Carte des Corrélations")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
        plt.title("Carte de corrélation entre les variables")
        st.pyplot(fig)

        # 💡 Analyse des Corrélations
        st.write("### 🔍 Analyse des Relations entre Variables")
        st.markdown(
            """
            - 🟢 **Corrélations fortes observées :**
              - `mean products sold` et `mean seller pass rate `
              - `total products liked` et `total bought`
            - 🟡 **Faibles corrélations détectées :**
              - `meanseniority`
              - `percentofiosusers`
            
            📌 **Interprétation :** Les variables similaires ont des liens forts entre elles, tandis que d'autres 
            (comme `meanseniority` et `percentofiosusers`) semblent moins liées aux autres variables du dataset.
            """
        )
