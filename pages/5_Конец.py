import streamlit as st
import time

with st.container(vertical_alignment="center", horizontal_alignment="center", height=400, border=False):
    st.markdown(f"<h1 style='text-align: center;'>Спасибо за внимание!</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center;'>(а теперь 3 вопроса...)</h2>", unsafe_allow_html=True)

while True:
    st.balloons()
    time.sleep(3)
