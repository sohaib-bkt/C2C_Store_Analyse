import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from slider import add_sidebar_links

add_sidebar_links()

# Titre principal
st.title("📊 Régression Ridge et Lasso")

# ============================
# Section 1 : Introduction
# ============================
st.header("📚 Introduction")
st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    🛠️ Les modèles Ridge et Lasso sont des techniques de régularisation utilisées pour améliorer les performances des modèles de régression linéaire, 
    en particulier lorsque les données présentent des problèmes de multicolinéarité (variables explicatives fortement corrélées) ou de surapprentissage (overfitting).
</p>
""", unsafe_allow_html=True)

# ============================
# Section 2 : Visualisation des Données
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

# ============================
# Section 3 : Préparation des Données
# ============================
st.header("🔧 Préparation des Données")

# Définir les fonctionnalités et la cible
features = ['totalproductslisted', 'totalbought', 'totalwished', 'totalproductsliked']
target = 'totalproductssold'

X = sellers[features]
y = sellers[target]

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ============================
# Section 4 : Régression Ridge
# ============================
st.header("🏔️ Régression Ridge")

# Initialisation du modèle Ridge
ridge = Ridge()

# Grille d'hyperparamètres pour Ridge
param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Recherche des meilleurs paramètres
grid_search_ridge = GridSearchCV(ridge, param_grid_ridge, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search_ridge.fit(X_train, y_train)

# Meilleurs paramètres pour Ridge
st.write(f"**Meilleurs paramètres pour Ridge :** {grid_search_ridge.best_params_}")

# Modèle Ridge optimisé
best_ridge = grid_search_ridge.best_estimator_

# Prédictions et évaluation
y_train_pred_ridge = best_ridge.predict(X_train)
y_test_pred_ridge = best_ridge.predict(X_test)

train_mse_ridge = mean_squared_error(y_train, y_train_pred_ridge)
test_mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)
train_r2_ridge = r2_score(y_train, y_train_pred_ridge)
test_r2_ridge = r2_score(y_test, y_test_pred_ridge)

st.subheader("📊 Performance du Modèle Ridge")
st.write(f"**MSE (Ensemble d'Entraînement) :** {train_mse_ridge:.2f}")
st.write(f"**MSE (Ensemble de Test) :** {test_mse_ridge:.2f}")
st.write(f"**R² (Ensemble d'Entraînement) :** {train_r2_ridge:.4f}")
st.write(f"**R² (Ensemble de Test) :** {test_r2_ridge:.4f}")

# ============================
# Section 5 : Régression Lasso
# ============================
st.header("🎯 Régression Lasso")

# Initialisation du modèle Lasso
lasso = Lasso()

# Grille d'hyperparamètres pour Lasso
param_grid_lasso = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Recherche des meilleurs paramètres
grid_search_lasso = GridSearchCV(lasso, param_grid_lasso, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search_lasso.fit(X_train, y_train)

# Meilleurs paramètres pour Lasso
st.write(f"**Meilleurs paramètres pour Lasso :** {grid_search_lasso.best_params_}")

# Modèle Lasso optimisé
best_lasso = grid_search_lasso.best_estimator_

# Prédictions et évaluation
y_train_pred_lasso = best_lasso.predict(X_train)
y_test_pred_lasso = best_lasso.predict(X_test)

train_mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)
train_r2_lasso = r2_score(y_train, y_train_pred_lasso)
test_r2_lasso = r2_score(y_test, y_test_pred_lasso)

st.subheader("📊 Performance du Modèle Lasso")
st.write(f"**MSE (Ensemble d'Entraînement) :** {train_mse_lasso:.2f}")
st.write(f"**MSE (Ensemble de Test) :** {test_mse_lasso:.2f}")
st.write(f"**R² (Ensemble d'Entraînement) :** {train_r2_lasso:.4f}")
st.write(f"**R² (Ensemble de Test) :** {test_r2_lasso:.4f}")

# ============================
# Section 6 : Visualisation des Coefficients
# ============================
st.header("📉 Visualisation des Coefficients")

# Créer des graphiques
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Coefficients Ridge
axes[0].barh(features, best_ridge.coef_)
axes[0].set_title("Coefficients Ridge")
axes[0].set_xlabel("Valeur du coefficient")

# Coefficients Lasso
axes[1].barh(features, best_lasso.coef_)
axes[1].set_title("Coefficients Lasso")
axes[1].set_xlabel("Valeur du coefficient")

plt.tight_layout()
st.pyplot(fig)

# ============================
# Section 7 : Comparaison des Modèles
# ============================
st.header("🆚 Comparaison des Modèles")
st.write(f"**Ridge Test MSE :** {test_mse_ridge:.2f} | **Lasso Test MSE :** {test_mse_lasso:.2f}")
st.write(f"**Ridge Test R² :** {test_r2_ridge:.4f} | **Lasso Test R² :** {test_r2_lasso:.4f}")

st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    🏆 Les modèles Ridge et Lasso montrent des performances exceptionnelles les données, avec une généralisation presque parfaite. 
    Ridge est légèrement meilleur en termes de MSE et R², mais Lasso offre une simplification du modèle en sélectionnant les variables les plus importantes. 
    Choisissez Ridge si la performance est prioritaire, ou Lasso si la simplicité et l'interprétabilité sont plus importantes.
</p>
""", unsafe_allow_html=True)



# Définir les fonctionnalités et la cible
features = ['totalproductslisted', 'totalbought', 'totalwished', 'totalproductsliked']
target = 'totalproductssold'

X = sellers[features]
y = sellers[target]

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ============================
# Section 3 : Régression Lasso
# ============================
st.header("🎯 Régression Lasso")

# Initialisation et entraînement du modèle Lasso
lasso = Lasso(alpha=0.1)  # Vous pouvez ajuster alpha selon vos besoins
lasso.fit(X_train, y_train)

# Prédictions avec Lasso
lasso_pred = lasso.predict(X_test)

# Graphique des valeurs réelles vs prédites
st.subheader("📈 Valeurs Réelles vs Prédites (Lasso)")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, lasso_pred, alpha=0.7, color='purple', label='Prédictions Lasso')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ligne de Référence')
ax.set_xlabel("Valeurs Réelles")
ax.set_ylabel("Valeurs Prédites")
ax.set_title("Valeurs Réelles vs Prédites (Lasso)")
ax.legend()
st.pyplot(fig)

# Interprétation
st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    📊 Le graphique montre une forte corrélation entre les valeurs réelles et prédites par le modèle Lasso. 
    La plupart des points sont proches de la ligne de référence, ce qui indique que le modèle fait des prédictions précises.
</p>
""", unsafe_allow_html=True)

# ============================
# Section 4 : Régression Ridge
# ============================
st.header("🏔️ Régression Ridge")

# Initialisation et entraînement du modèle Ridge
ridge = Ridge(alpha=1.0)  # Vous pouvez ajuster alpha selon vos besoins
ridge.fit(X_train, y_train)

# Prédictions avec Ridge
ridge_pred = ridge.predict(X_test)

# Graphique des valeurs réelles vs prédites
st.subheader("📈 Valeurs Réelles vs Prédites (Ridge)")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, ridge_pred, alpha=0.7, color='blue', label='Prédictions Ridge')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ligne de Référence')
ax.set_xlabel("Valeurs Réelles")
ax.set_ylabel("Valeurs Prédites")
ax.set_title("Valeurs Réelles vs Prédites (Ridge)")
ax.legend()
st.pyplot(fig)

# Interprétation
st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    📊 Le graphique montre une forte corrélation entre les valeurs réelles et prédites par le modèle Ridge. 
    Comme pour Lasso, la plupart des points sont proches de la ligne de référence, ce qui indique que le modèle fait des prédictions précises.
</p>
""", unsafe_allow_html=True)

# ============================
# Section 5 : Prédiction avec de Nouvelles Données
# ============================
st.header("🔮 Prédiction avec de Nouvelles Données")

# Exemple de nouvelles données
new_data = {
    'totalproductslisted': [150],  
    'totalbought': [75],           
    'totalwished': [50],           
    'totalproductsliked': [100]    
}

new_data = pd.DataFrame(new_data)
st.write("**Nouvelles Données :**")
st.write(new_data)
# Conversion en DataFrame
new_df = pd.DataFrame(new_data)

# Standardisation des nouvelles données
new_scaled = scaler.transform(new_df)

# Prédiction avec Lasso
new_pred_lasso = lasso.predict(new_scaled)
st.write(f"**Prédiction Lasso :** {new_pred_lasso[0]:.2f} produits vendus")

# Interprétation
st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    🎯 Le modèle de régression Lasso prédit qu'avec les caractéristiques fournies (produits listés, achetés, souhaités et aimés), environ 230 produits seront vendus.
</p>
""", unsafe_allow_html=True)

# Prédiction avec Ridge
new_pred_ridge = ridge.predict(new_scaled)
st.write(f"**Prédiction Ridge :** {new_pred_ridge[0]:.2f} produits vendus")

# Interprétation
st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    🎯 Le modèle de régression Ridge prédit qu'avec les caractéristiques fournies (produits listés, achetés, souhaités et aimés), environ 168 produits seront vendus.
</p>
""", unsafe_allow_html=True)