import streamlit as st
import numpy as np

def exec_and_return(code, ret_vals):
    ld = {}
    exec(code, globals(), ld)
    
    if isinstance(ret_vals, str):
        return ld.get(ret_vals)
    elif isinstance(ret_vals, list):
        return tuple(ld.get(val) for val in ret_vals)
    else:
        raise ValueError("ret_vals должен быть строкой или списком строк")

def define_function(header, description, doc = None, example = None, header_code = False, code = None):
    if header_code:
        st.code(header, language='python')
    else:
        st.header(header, anchor = False)

    if code:
        st.code(code, language = 'python')

    st.write(description) 
    
    if doc:
        url = f"https://numpy.org/doc/stable/reference/generated/{doc}.html"
        st.markdown(f"[Подробнее]({url})")

    if example:
        try:
            example()
        except Exception as e:
            st.error(f"Ошибка: {e}")
