import streamlit as st

st.balloons()
with st.container(vertical_alignment="center", horizontal_alignment="center", height=300, border=False):
    st.markdown(f"<h1 style='text-align: center;'>Спасибо за внимание!</h1>", unsafe_allow_html=True)
