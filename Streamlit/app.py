import streamlit as st
import subprocess
from slider import add_sidebar_links 

# Configuration de la page
st.set_page_config(page_title="📊 Analyse du Comportement des Utilisateurs", layout="wide")
st.sidebar.empty()

# Titre principal
st.title("👋 Bienvenue dans l'Analyse du Comportement des Utilisateurs")

# Ajouter les liens de la barre latérale
add_sidebar_links()


st.header("🔍 Analyse du Comportement des Utilisateurs")
st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    🛒 Dans une plateforme C2C, les utilisateurs jouent un rôle crucial en tant que fournisseurs et clients. 
    Comprendre leur comportement est essentiel pour optimiser l'engagement et la croissance.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    📊 Ce jeu de données, extrait d'une boutique de mode en ligne lancée en Europe en 2009, après environ 10 ans 
    de lancement du site Web, se concentre sur les utilisateurs enregistrés actifs.
</p>
""", unsafe_allow_html=True)

st.subheader("🎯 Objectif Principal")
st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    Analyser et visualiser le comportement des utilisateurs afin d'identifier les facteurs influençant leur 
    activité et leur engagement sur la plateforme.
</p>
""", unsafe_allow_html=True)

st.subheader("📋 Hypothèses")
st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    1. 📱 L'utilisation d'une application mobile influence l'engagement des utilisateurs, notamment en termes d'achats et de ventes.<br>
    2. 🌐 Les utilisateurs ayant une présence sociale plus forte (suiveurs, likes, etc.) sont plus actifs en termes de transactions.<br>
    3. 👥 Certains profils démographiques (genre, ancienneté, langue) ont un impact sur l'activité des utilisateurs.<br>
    4. ⏳ L'ancienneté sur la plateforme et la fréquence de connexion influencent la fidélisation et l'activité des utilisateurs.
</p>
""", unsafe_allow_html=True)

st.subheader("📸 Illustration")
col1, col2, col3 = st.columns([1, 2, 1])  # Créer trois colonnes
with col2:  # Placer l'image dans la colonne centrale
    st.image("image.jpg", use_container_width=True)  # Adapter l'image à la largeur du conteneur


st.subheader("🚀 Exécuter l'Application de Prédiction")
col1, col2, col3 = st.columns([2.5, 2, 1])  # Créer trois colonnes
with col2:  # Placer le bouton dans la colonne centrale
    if st.button('▶️ Lancer l\'Application de Prédiction', help="Cliquez pour exécuter l'application de prédiction"):
        try:
            subprocess.run(["streamlit", "run", "predir_app.py"], check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Une erreur s'est produite : {e}")