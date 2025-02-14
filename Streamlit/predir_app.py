import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from slider import add_sidebar_links

add_sidebar_links()

# 🎯 Charger les modèles et le scaler
ridge_model = pickle.load(open('models/ridge_model.pkl', 'rb'))
lasso_model = pickle.load(open('models/lasso_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Titre de l'application
st.title('📊 Prédiction des Produits Vendus')

# 📝 Saisie des informations par l'utilisateur
st.subheader("🔢 Entrez les informations suivantes :")
totalproductslisted = st.number_input("📦 Nombre total de produits listés", min_value=0)
totalbought = st.number_input("🛒 Nombre total de produits achetés", min_value=0)
totalwished = st.number_input("💭 Nombre total de produits souhaités", min_value=0)
totalproductsliked = st.number_input("❤️ Nombre total de produits aimés", min_value=0)

# 🔮 Bouton de prédiction
if st.button('🔮 Prédire'):
    # Préparer les données d'entrée dans le même format que les données d'entraînement
    input_data = pd.DataFrame({
        'totalproductslisted': [totalproductslisted],
        'totalbought': [totalbought],
        'totalwished': [totalwished],
        'totalproductsliked': [totalproductsliked]
    })
    
    # 🌟 Normalisation des données d'entrée
    input_scaled = scaler.transform(input_data)

    # 📈 Obtenir les prédictions des deux modèles
    ridge_prediction = ridge_model.predict(input_scaled)
    lasso_prediction = lasso_model.predict(input_scaled)
    
    # 💡 Affichage des prédictions
    st.write(f"🔹 **Prédiction avec la régression Ridge :** {ridge_prediction[0]:.2f} produits vendus")
    st.write(f"🔸 **Prédiction avec la régression Lasso :** {lasso_prediction[0]:.2f} produits vendus")

