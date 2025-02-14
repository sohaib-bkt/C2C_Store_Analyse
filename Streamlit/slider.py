import streamlit as st

def add_sidebar_links():
    st.sidebar.empty()
    st.sidebar.title("📊 Data Analysis")
    st.sidebar.page_link("pages/1_comprehension.py", label="Data Comprehension")
    st.sidebar.page_link("pages/2_nettoyage.py", label="Data Cleaning")
    st.sidebar.page_link("pages/3_visualisation.py", label="Data Visualization I")
    st.sidebar.page_link("pages/4_transformation.py", label="Transformation des donnees")
    st.sidebar.page_link("pages/5_visualisation2.py", label="Data Visualization II")

    st.sidebar.title("📚 Classification")
    st.sidebar.page_link("pages/7_cah.py", label="CAH (Hierarchical Clustering)")
    st.sidebar.page_link("pages/6_sellersData.py", label="Seller Data ML")
    st.sidebar.page_link("pages/8_Kmeans.py", label="K-Means Clustering")


    st.sidebar.title("🤖 Prediction")   
    st.sidebar.page_link("pages/9_regressionLinear.py", label="Regression Linear")
    st.sidebar.page_link("pages/10_regressionRidgeLasso.py", label="Regression Ridge/Lasso")

    st.sidebar.title("📌 Conclusion")
    st.sidebar.page_link("pages/11_coclusion.py", label="Conclusion")

    