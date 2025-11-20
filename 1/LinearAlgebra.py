import streamlit as st
import sys
from PIL import Image

sys.path.append('.')

col1, col2 = st.columns([6, 1], vertical_alignment="center")

with col1:
    st.markdown(f"<h2 style='text-align: center;'>Филиал МГУ имени М.В. Ломоносова, г. Саров</h2>", unsafe_allow_html=True)

with col2:
    image = Image.open('./logo.png') 
    st.image(image)

st.markdown(f"<br/><br/><h1 style='text-align: center;'>Численные методы решения задач линейной алгебры</h1>", unsafe_allow_html=True)

r"""
\
\
Выполнила группа №1:
* Банников Николай Владимирович
* Быканова Ульяна Федоровна
* Василенко Даниил Олегович
* Гайворонская Юлия Евгеньевна
* Демаков Матвей Александрович
"""

st.markdown(f"<p style='text-align: center; font-size: 14pt;'>2025 год</p>", unsafe_allow_html=True)
