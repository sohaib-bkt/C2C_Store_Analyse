import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from slider import add_sidebar_links
from geopy.geocoders import Nominatim
import plotly.express as px
import time

# Configuration de la page
st.set_page_config(page_title="📊 Tableau de Bord des Données Utilisateurs", layout="wide")

# Titre de l'application
st.title("📊 Tableau de Bord des Données Utilisateurs")
add_sidebar_links()

# Chemin du fichier de données
file_path = "data/user_data.csv"

# Vérification de l'existence du fichier
if not os.path.exists(file_path):
    st.error(f"Le fichier {file_path} n'existe pas.")
else:
    db = pd.read_csv(file_path)

    # Section 1: Distribution des jours depuis la dernière connexion
    st.header("📅 Distribution des jours depuis la dernière connexion")
    plt.figure(figsize=(10, 6))
    sns.histplot(db['daysSinceLastLogin'], bins=30, kde=True, color='skyblue')
    plt.axvline(x=np.mean(db['daysSinceLastLogin']), color='red', linestyle='--', label='Moyenne')
    plt.title('Distribution des jours depuis la dernière connexion')
    plt.xlabel('Jours')
    plt.ylabel('Fréquence')
    plt.legend()
    st.pyplot(plt)
    plt.clf()
    st.markdown("""
    <p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
        🧐 La majorité des utilisateurs se connectent fréquemment, tandis qu’un groupe restreint reste inactif pendant de longues périodes.
    </p>
    """, unsafe_allow_html=True)

    # Section 2: Agrégation des données par 'hasAnyApp'
    st.header("📱 Agrégation des données par 'hasAnyApp'")
    df_group_two = db[['hasAnyApp', 'productsBought', 'productsSold']].groupby(['hasAnyApp']).agg([np.sum, np.mean])
    st.dataframe(df_group_two.style.background_gradient(cmap='Blues'))
    st.markdown("""
    <p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
        📈 Les utilisateurs ayant une application achètent et vendent significativement plus de produits par rapport à ceux qui n'en ont pas.
    </p>
    """, unsafe_allow_html=True)

    # Section 3: Utilisateurs ayant utilisé une application officielle
    st.header("📲 Utilisateurs ayant utilisé une application officielle")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='hasAnyApp', data=db, palette='viridis')
    plt.title('Utilisateurs ayant utilisé une application officielle')
    st.pyplot(plt)
    plt.clf()
    st.markdown("""
    <p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
        📉 La plupart des utilisateurs n'ont pas utilisé l'application mobile.
    </p>
    """, unsafe_allow_html=True)

    # Section 4: Agrégation des données par 'civilityTitle'
    st.header("👤 Agrégation des données par 'civilityTitle'")
    df_group_one = db[['civilityTitle', 'productsBought', 'productsSold']].groupby(['civilityTitle']).agg([np.sum, np.mean])
    st.dataframe(df_group_one.style.background_gradient(cmap='Greens'))
    st.markdown("""
    <p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
        👨‍👩‍👧 Les hommes (mr) et les dames mariées (mrs) semblent plus actifs dans les achats en termes de quantités totales. Toutefois, les mademoiselles montrent une moyenne d'achats et de ventes plus élevée par utilisateur.
    </p>
    """, unsafe_allow_html=True)

    # Titre de l'application
    st.title("📍 Carte de Densité des Utilisateurs par Pays")

    @st.cache_data
    def load_data():
        return pd.read_csv(file_path)

    db = load_data()

    # Interface utilisateur
    st.sidebar.header("Options")
    sample_frac = st.sidebar.slider("Échantillon (%)", 0.0001, 1.0, 0.01, format="%.4f")

    if st.sidebar.button("📌 Générer la Carte"):
        st.write("⏳ Traitement des données en cours...")

        # Échantillonnage des données
        db_part = db.sample(frac=sample_frac).copy()

        # Géolocalisation
        geolocator = Nominatim(user_agent="geoapi")

        def get_lat_lon(country):
            try:
                location = geolocator.geocode(country)
                if location:
                    return location.latitude, location.longitude
                else:
                    return None, None
            except Exception as e:
                st.warning(f"Erreur lors de la géolocalisation pour {country}: {e}")
                return None, None

        # Progress bar
        progress_bar = st.progress(0)
        latitudes, longitudes = [], []

        for i, country in enumerate(db_part["country"]):
            lat, lon = get_lat_lon(country)
            latitudes.append(lat)
            longitudes.append(lon)
            progress_bar.progress((i + 1) / len(db_part))
            time.sleep(1)  # Évite les limites de requêtes

        db_part["lat"], db_part["lon"] = latitudes, longitudes

        # Supprimer les valeurs nulles
        db_part.dropna(subset=["lat", "lon"], inplace=True)

        # Calculer la densité par pays
        country_density = db_part["country"].value_counts().reset_index()
        country_density.columns = ["country", "density"]

        # Fusionner les données
        db_part = db_part.drop_duplicates(subset=["country"])
        db_part = db_part.merge(country_density, on="country")

        # Création de la carte interactive
        fig = px.scatter_mapbox(
            db_part,
            lat="lat",
            lon="lon",
            size="density",
            color="density",
            hover_name="country",
            color_continuous_scale="Viridis",
            mapbox_style="open-street-map",
            zoom=1
        )

        # Affichage de la carte
        st.plotly_chart(fig)

    # Section 5: Distribution des titres de civilité
    st.header("👥 Distribution des titres de civilité")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='civilityTitle', data=db, palette='magma')
    plt.title('Distribution des titres de civilité')
    st.pyplot(plt)
    plt.clf()
    st.markdown("""
    <p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
        👩‍💼 Les femmes mariées semblent être les utilisatrices les plus actives de ce site.
    </p>
    """, unsafe_allow_html=True)

    # Section 6: Langues utilisées sur le site
    st.header("🌍 Langues utilisées sur le site")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='language', data=db, palette='coolwarm')
    plt.title('Langues utilisées sur le site')
    st.pyplot(plt)
    plt.clf()

    st.header("🌐 Agrégation des données par 'language'")
    df_group_three = db[['language', 'productsBought', 'productsSold']].groupby(['language']).agg([np.sum, np.mean])
    st.dataframe(df_group_three.style.background_gradient(cmap='Purples'))
    st.markdown("""
    <p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
        🗣️ La première langue préférée des utilisateurs est l'anglais et la seconde le français. Cependant, ils n'ont pas la moyenne de produits achetés et de produits vendus la plus élevée.
    </p>
    """, unsafe_allow_html=True)

    # Section 7: Répartition des langues par titre de civilité
    st.header("🌐 Répartition des langues par titre de civilité")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='language', data=db, hue='civilityTitle', palette='Set2')
    plt.title('Répartition des langues par titre de civilité')
    st.pyplot(plt)
    plt.clf()
    st.markdown("""
    <p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
        🌍 Le countplot nous indique qu'il y a plus d'utilisatrices mariées dans le monde.
    </p>
    """, unsafe_allow_html=True)


    # Section 8: Corrélation entre les variables
    st.header("🔍 Corrélation entre les variables")
    file_path_encoded = "data/user_data_encoded.csv"
    if os.path.exists(file_path_encoded):
        db2 = pd.read_csv(file_path_encoded)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(db2.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        plt.clf()
        st.markdown("""
        <p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
            🔗 Variable <b>daysSinceLastLogin & hasProfilePicture</b> : semblent être négativement corrélées à toutes les autres variables. Cependant, seuls 1,95 % des utilisateurs n’avaient pas de photo de profil.
            <br><br>
            🔗 Variables <b>socialNbFollowers & socialNBFollows & socialproductsLiked & productsListed & productsSold & productsPassRate & productsWished</b> semblent être positivement liés les uns aux autres.
            <br><br>
            🔗 La variable <b>seniority</b> ancienneté semble décorrélée à toutes les autres variables.
            <br><br>
            🔗 Variables <b>language et country</b> semblent être négativement corrélés, ont une corrélation hebdomadaire avec les variables hasAnyApp & hasIosApp & hasAndroidApp ; mais n'ont presque aucune corrélation avec d'autres variables. Pour l’instant, nous pourrions conserver ces variables pour une analyse plus approfondie.
        </p>
        """, unsafe_allow_html=True)

        # Supprimer les variables sans corrélation
        no_columns = ['seniority', 'seniorityAsMonths', 'seniorityAsYears']
        db3 = db2.drop(no_columns, axis=1)
        # Section 8.1: Corrélation entre 'socialProductsLiked' et 'productsBought'
        st.header("📊 Corrélation entre 'socialProductsLiked' et 'productsBought'")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(db3[['socialProductsLiked', 'productsBought']].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        plt.title("Corrélation entre 'socialProductsLiked' et 'productsBought'")
        st.pyplot(fig)
        plt.clf()
        st.markdown("""
        <p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
            🔍 Cela signifie que le nombre de produits aimés sur les réseaux sociaux n'a presque aucun lien avec les achats effectués.
        </p>
        """, unsafe_allow_html=True)

        # Section 9: Segmentation des activités
        st.header("📊 Segmentation des activités en fonction des jours depuis la dernière connexion")
        bins = [-1, 30, 120, np.inf]
        labels = ['Active (0-30 days)', 'Moderate (30-120 days)', 'Dormant (120+ days)']
        db3 = db2.copy()
        db3['activity_tier'] = pd.cut(db3['daysSinceLastLogin'], bins=bins, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        db3['activity_tier'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'orange', 'lightcoral'], ax=ax)
        st.pyplot(fig)
        plt.clf()
        st.markdown("""
        <p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
            📉 Cela indique un engagement utilisateur faible ou une rétention insuffisante. Une stratégie de réactivation pourrait être nécessaire pour améliorer l'engagement.
        </p>
        """, unsafe_allow_html=True)

        # Section 10: Préférences de plateforme
        st.header("📱 Préférences de plateforme")
        platform_stats = db3.agg({
            'hasAnyApp_encoded': 'mean',
            'hasAndroidApp_encoded': 'mean',
            'hasIosApp_encoded': 'mean'
        })
        fig, ax = plt.subplots(figsize=(8, 6))
        platform_stats.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'], ax=ax)
        st.pyplot(fig)
        plt.clf()
        st.markdown("""
        <p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
            📲 La majorité des utilisateurs n'ont pas utilisé l'application mobile, tandis que ceux qui l'ont utilisée préfèrent iOS à Android.
        </p>
        """, unsafe_allow_html=True)

        # Section 11: Analyse des utilisateurs avec ou sans photo de profil
        st.header("📸 Analyse des utilisateurs avec ou sans photo de profil")
        profile_analysis = db3.groupby('hasProfilePicture_encoded').agg({
            'socialNbFollowers': 'mean',
            'productsSold': 'mean',
            'productsBought': 'mean'
        }).reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        profile_analysis.plot(kind='bar', x='hasProfilePicture_encoded', color=['skyblue', 'lightgreen', 'salmon'], ax=ax)
        st.pyplot(fig)
        plt.clf()
        st.markdown("""
        <p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
            📷 La majorité des utilisateurs n'ont pas de photo de profil, et font les plus achats.
        </p>
        """, unsafe_allow_html=True)

