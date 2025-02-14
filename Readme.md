# 📊 Projet Streamlit - Analyse de Données

## 🚀 Description
Ce projet est une application Streamlit permettant d'analyser et de visualiser des données avec des graphiques interactifs et une carte de densité par pays.

## 📂 Structure du projet
- `Streamlit/app.py` : Fichier principal contenant le code de l'application Streamlit.
- `Streamlit/slider.py` : Module pour ajouter des liens et fonctionnalités supplémentaires à la barre latérale.
- `Streamlit/requirements.txt` : Liste des dépendances nécessaires.
- `Jupyter/Main.ipynb` : Le notebook contient les analyses .
- `data` : Le répertoire contenant les tables nécessaires à l'analyse dans Jupyter .

## 🛠 Fonctionnalités
✅ Chargement et exploration des données  
✅ Échantillonnage des données via un slider ajustable  
✅ Géolocalisation automatique des pays à l'aide de `geopy`  
✅ Affichage de graphiques interactifs avec Matplotlib, Seaborn et Plotly  
✅ Carte interactive de densité avec Plotly ou Folium  
✅ Optimisation des performances avec `@st.cache_data` et `st.session_state`
✅ Faire des prediction grace au regression `Linear` et `Ridge/Lasso`
✅ Faire des classification avec `Kmeans` et `CAH`



## 🔧 Installation et utilisation

### 1️⃣ Installer les dépendances
Assurez-vous d'avoir Python 3.8+ installé, puis exécutez :
```bash
pip install -r requirements.txt
```

### 2️⃣ Lancer l'application
```bash
streamlit run app.py
```
   