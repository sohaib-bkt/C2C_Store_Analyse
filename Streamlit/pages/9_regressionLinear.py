import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from slider import add_sidebar_links

add_sidebar_links()

# Titre principal
st.title("📈 Analyse des Vendeurs et Modélisation")

# ============================
# Section 1 : Visualisation des Données
# ============================
st.header("📊 Visualisation des Données")

# Charger les données (remplacez par votre propre dataset)
sellers = pd.read_csv("data/seller.csv")  # Assurez-vous d'avoir un fichier CSV approprié

# Créer des graphiques
st.subheader("🔍 Scatter Plots des Vendeurs")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Graphique 1 : Total Products Sold vs Total Products Listed
axes[0, 0].scatter(sellers.totalproductssold, sellers.totalproductslisted, alpha=0.6, color='blue')
axes[0, 0].set_xlabel('Total Products Sold')
axes[0, 0].set_ylabel('Total Products Listed')
axes[0, 0].set_title('Total Products Sold vs Total Products Listed')

# Graphique 2 : Total Products Sold vs Total Products Bought
axes[0, 1].scatter(sellers.totalproductssold, sellers.totalbought, alpha=0.6, color='green')
axes[0, 1].set_xlabel('Total Products Sold')
axes[0, 1].set_ylabel('Total Products Bought')
axes[0, 1].set_title('Total Products Sold vs Total Products Bought')

# Graphique 3 : Total Products Sold vs Total Products Wished
axes[1, 0].scatter(sellers.totalproductssold, sellers.totalwished, alpha=0.6, color='orange')
axes[1, 0].set_xlabel('Total Products Sold')
axes[1, 0].set_ylabel('Total Products Wished')
axes[1, 0].set_title('Total Products Sold vs Total Products Wished')

# Graphique 4 : Total Products Sold vs Total Products Liked
axes[1, 1].scatter(sellers.totalproductssold, sellers.totalproductsliked, alpha=0.6, color='red')
axes[1, 1].set_xlabel('Total Products Sold')
axes[1, 1].set_ylabel('Total Products Liked')
axes[1, 1].set_title('Total Products Sold vs Total Products Liked')

plt.tight_layout()
st.pyplot(fig)

# Explication des graphiques
st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    📈 Ces scatter plots montrent une corrélation positive entre le nombre total de produits vendus et les autres variables (listés, achetés, souhaités, aimés). 
    Quelques points extrêmes indiquent des vendeurs très actifs qui influencent fortement les tendances. 
    La densité élevée près de l'origine suggère que la majorité des vendeurs ont une activité limitée. 
    Plus un vendeur vend, plus ses produits sont listés, achetés, souhaités et aimés.
</p>
""", unsafe_allow_html=True)

# ============================
# Section 2 : Standardisation et ACP
# ============================
st.header("🔧 Standardisation et Analyse en Composantes Principales (ACP)")

# Standardisation des variables
features = ['totalproductslisted', 'totalbought', 'totalwished', 'totalproductsliked']
target = 'totalproductssold'

X = sellers[features]
y = sellers[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ACP
pca_temp = PCA().fit(X_scaled)
cumulative_variance = pca_temp.explained_variance_ratio_.cumsum()
st.write(f"**Variance Cumulée Expliquée :** {cumulative_variance}")

n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

st.write(f"**Variance Expliquée par Composant :** {pca.explained_variance_ratio_}")

st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    🎯 Les deux premiers composants principaux expliquent 99.7% de la variance cumulée des données, ce qui signifie qu'ils capturent presque toute l'information pertinente. 
    Le premier composant seul explique 98.5% de la variance, tandis que le deuxième n'ajoute que 1.1%. 
    Cela suggère que la réduction à deux dimensions est très efficace pour représenter les données.
</p>
""", unsafe_allow_html=True)

# ============================
# Section 3 : Modélisation
# ============================
st.header("🤖 Modélisation avec Régression Linéaire")

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Entraînement du modèle
lr = LinearRegression()
lr.fit(X_train, y_train)

# Prédictions
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Métriques d'évaluation
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

st.subheader("📊 Performance du Modèle")
st.write(f"**MSE (Ensemble d'Entraînement) :** {train_mse}")
st.write(f"**MSE (Ensemble de Test) :** {test_mse}")
st.write(f"**R² (Ensemble d'Entraînement) :** {train_r2}")
st.write(f"**R² (Ensemble de Test) :** {test_r2}")

# Validation croisée
cv_scores = cross_val_score(lr, X_pca, y, cv=5, scoring='neg_mean_squared_error')
st.write(f"**MSE Moyen (Validation Croisée) :** {-cv_scores.mean():.2f}")

st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    🚨 Le MSE moyen obtenu par validation croisée est de 134 442,33, ce qui est nettement plus élevé que le MSE sur le test. 
    Cela pourrait indiquer que le modèle est moins performant sur certains sous-ensembles de données, ce qui renforce l'hypothèse d'un possible surapprentissage.
</p>
""", unsafe_allow_html=True)

# Graphique des valeurs réelles vs prédites
st.subheader("📈 Valeurs Réelles vs Prédites")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_test_pred, alpha=0.7, color='purple')
ax.set_xlabel("Valeurs Réelles")
ax.set_ylabel("Valeurs Prédites")
ax.set_title("Valeurs Réelles vs Prédites")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
st.pyplot(fig)

st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    ✅ Presque tous les points suivent cette ligne, le modèle fait des prédictions très précises. 
    Cela suggère une forte corrélation entre les valeurs réelles et prédites.
</p>
""", unsafe_allow_html=True)

# ============================
# Section 4 : Prédiction avec de Nouvelles Données
# ============================
st.header("🔮 Prédiction avec de Nouvelles Données")

# Exemple de nouvelles données
new_data = {
    'totalproductslisted': [150],   # Exemple de valeur pour le total des produits répertoriés
    'totalbought': [75],            # Exemple de valeur pour le total des produits achetés
    'totalwished': [50],            # Exemple de valeur pour le total des produits souhaités
    'totalproductsliked': [100]     # Exemple de valeur pour le total des produits aimés
}

# Convertir les données en DataFrame
st.write("**Nouvelles Données pour Prédiction :**")
new_df = pd.DataFrame(new_data)
st.write(new_df)

# Conversion en DataFrame
new_df = pd.DataFrame(new_data)

# Standardisation et ACP
new_scaled = scaler.transform(new_df)
new_pca = pca.transform(new_scaled)

# Prédiction
predicted_products_sold = lr.predict(new_pca)
st.write(f"**Total des Produits Vendus Prédits :** {predicted_products_sold[0]:.2f}")