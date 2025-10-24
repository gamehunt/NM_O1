import streamlit as st
import numpy as np

from shared import *

# st.title("Работа с полиномами в NumPy")

def example():
    code = st.text_area("Интерактивный код:",
value="""
from numpy import polynomial
# Создать полином:

coef = [2, 3, 1]  
p = polynomial.Polynomial(coef)

#Значения полинома p в точках x:
x = np.array([0, 1, 2])
values = p(x)

#Сумма двух полиномов:
q = polynomial.Polynomial([1, 4])
sum_poly = p + q

# Произведение полиномов p и q:
product_poly = p * q

# Корни полинома p:
roots = np.roots(coef)

diff = p.deriv()
integ = p.integ()
""", height=400)

    if st.button("Запустить"):
            p, values, sum_poly, product_poly, roots, diff, integ = exec_and_return(code, ["p", "values", "sum_poly", "product_poly", "roots", "diff", "integ"])

            st.write("Полином p(x):")
            st.write(p)
            st.write("#### Значения полинома p в точках x:")
            st.write(values)
            st.write("### Сумма полиномов p и q:")
            st.write(sum_poly)
            st.write("### Произведение полиномов p и q:")
            st.write(product_poly)
            st.write("### Корни полинома p:")
            st.write(roots)
            st.write("### Производная полинома p:")
            st.write(diff)
            st.write("### Интеграл от полинома p:")
            st.write(integ)
    
define_function("Работа с полиномами в NumPy",
                   """Работа с полиномами в NumPy осуществляется с помощью модуля :red[numpy.polynomial].\n
Помимо обычных многочленов, модуль поддерживает работу со следующими видами полиномов:
* Многочлены Чебышева (:red[numpy.polynomial.chebyshev])
* Многочлены Лежандра (:red[numpy.polynomial.legendre])
* Многочлены Лаггера (:red[numpy.polynomial.laguerre])
* Многочлены Эрмита (:red[numpy.polynomial.hermite])
                   """,
                   doc="../routines.polynomials",
                   example=example,
                   code="numpy.polynomial.polynomial.Polynomial(coef, domain=None, window=None, symbol='x')\nnumpy.roots(p)")
