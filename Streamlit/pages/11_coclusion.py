import streamlit as st
from slider import add_sidebar_links

add_sidebar_links()


st.subheader("📌 Conclusion")
st.image("concc.jpg")
st.markdown("""
<p style="text-align: justify; background-color: #e8f4f8; padding: 10px; border-radius: 10px; color: #333;">
    🔍 L'analyse du comportement des utilisateurs sur cette plateforme C2C révèle des tendances essentielles pour optimiser l'engagement. 
    📱 L'utilisation d'une application mobile favorise une activité accrue, tandis que la présence sociale joue un rôle clé dans le volume des transactions. 
    🌍 De plus, certains facteurs démographiques, tels que l'ancienneté et la langue, influencent la fidélité des utilisateurs. 
    ⏳ L'ancienneté sur la plateforme est également un indicateur de rétention et d'engagement. 
    📈 Ainsi, pour maximiser les interactions et la croissance, il est crucial d'améliorer l'expérience mobile, d'encourager l'engagement social et d'adapter les stratégies de fidélisation en fonction des profils des utilisateurs. 
    🚀 En intégrant ces insights, la plateforme pourra renforcer sa communauté et stimuler l'activité commerciale de manière durable.
</p>
""", unsafe_allow_html=True)
