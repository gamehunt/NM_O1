import streamlit as st
import numpy as np

from shared import *

st.title("Функции линейной алгебры в NumPy")

# Меню
menu = st.sidebar.radio("***",
                        (
                            "numpy.linalg.inv",
                            "numpy.linalg.det",
                            "numpy.linalg.eig",
                            "numpy.linalg.solve",
                            "numpy.linalg.norm",
                            "numpy.linalg.svd",
                            "numpy.linalg.cholesky",
                            "numpy.linalg.pinv",
                            "numpy.linalg.qr",
                            "numpy.linalg.eigh",
                            "numpy.linalg.lstsq",
                            "numpy.polynomial",
                        )
                        )

# numpy.linalg.inv
if menu == "numpy.linalg.inv":
    def example():
        code = st.text_area("Введите код для преобразования массива:",
                            value="A = np.array([[3, 2], [3, 4]])\ninv_A = np.linalg.inv(A)", height=100)
        if st.button("Вычислить обратную матрицу"):
            inv_A, A = exec_and_return(code, ["inv_A", "A"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Матрица A:")
                st.write(A)
            with col2:
                if A.shape[0] == A.shape[1]:
                    st.write("Обратная матрица A:")
                    st.write(inv_A)
                else:
                    st.write("Матрица A не является квадратной, обратная матрица не существует.")
    
    define_function("numpy.linalg.inv",
                   """Находит обратную матрицу для квадратной матрицы. 
                   Обратная матрица A обозначается как A^(-1) и удовлетворяет уравнению A * A^(-1) = I, 
                   где I — единичная матрица. Если матрица не является невырожденной (определитель равен нулю), 
                   то обратная матрица не существует.""",
                   doc="numpy.linalg.inv",
                   example=example)

# numpy.linalg.det
if menu == "numpy.linalg.det":
    def example():
        code = st.text_area("Введите код для преобразования массива:",
                            value="A = np.array([[3, 2], [3, 4]])\ndet_A = np.linalg.det(A)", height=100)
        if st.button("Вычислить определитель"):
            det_A, A = exec_and_return(code, ["det_A", "A"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Матрица A:")
                st.write(A)
            with col2:
                if A.shape[0] == A.shape[1]:
                    st.write("Определитель матрицы A:")
                    st.write(det_A)
                else:
                    st.write("Определитель можно вычислить только для квадратной матрицы.")
    
    define_function("numpy.linalg.det",
                   """Вычисляет определитель матрицы, который является скалярным значением, 
                   характеризующим свойства матрицы. Определитель используется для проверки, является ли 
                   матрица вырожденной (определитель равен нулю) и в решении систем линейных уравнений.""",
                   doc="numpy.linalg.det",
                   example=example)

# numpy.linalg.eig
if menu == "numpy.linalg.eig":
    def example():
        code = st.text_area("Введите код для преобразования массива:",
                            value="A = np.array([[3, 2], [3, 4]])\neigenvalues, eigenvectors = np.linalg.eig(A)", height=100)
        if st.button("Вычислить собственные вектора"):
            eigenvalues, eigenvectors, A = exec_and_return(code, ["eigenvalues", "eigenvectors", "A"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Матрица A:")
                st.write(A)
            with col2:
                if A.shape[0] == A.shape[1]:
                    st.write("Собственные значения:")
                    st.write(eigenvalues)
                else:
                    st.write("Собственные значения и векторы можно вычислить только для квадратной матрицы.")
            with col3:
                if A.shape[0] == A.shape[1]:
                    st.write("Собственные векторы:")
                    st.write(eigenvectors)
    
    define_function("numpy.linalg.eig",
                   """Находит собственные значения и собственные векторы матрицы. 
                   Собственные значения — это скаляры, которые показывают, как матрица растягивает 
                   или сжимает пространство, а собственные векторы — это направления, которые остаются 
                   неизменными при применении матрицы.""",
                   doc="numpy.linalg.eig",
                   example=example)

# numpy.linalg.solve
if menu == "numpy.linalg.solve":
    def example():
        code = st.text_area("Введите код для преобразования массива:",
                            value="A = np.array([[3, 2], [3, 4]])\nb = np.array([5, 11])\nx = np.linalg.solve(A, b)", height=100)
        if st.button("Вычислить СЛАУ"):
            x, A, b = exec_and_return(code, ["x", "A", "b"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Матрица A:")
                st.write(A)
            with col2:
                st.write("Вектор b:")
                st.write(b)
            with col3:
                if A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0]:
                    st.write("Решение системы:")
                    st.write(x)
                else:
                    st.write("Система уравнений не может быть решена, так как размеры матрицы A и вектора b не совпадают.")
    
    define_function("numpy.linalg.solve",
                   """Решает систему линейных уравнений вида Ax = b, где A — матрица коэффициентов, 
                   x — вектор переменных, а b — вектор свободных членов. Использует метод Гаусса для нахождения 
                   решения, если оно существует.""",
                   doc="numpy.linalg.solve",
                   example=example)

# numpy.linalg.norm
if menu == "numpy.linalg.norm":
    def example():
        code = st.text_area("Введите код для преобразования массива:",
                            value="A = np.array([[3, 2], [3, 4]])\nnorm_A = np.linalg.norm(A)", height=100)
        if st.button("Вычислить норму"):
            norm_A, A = exec_and_return(code, ["norm_A", "A"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Матрица A:")
                st.write(A)
            with col2:
                st.write("Норма матрицы A:")
                st.write(norm_A)
    
    define_function("numpy.linalg.norm",
                   """Вычисляет норму вектора или матрицы, которая является мерой их "размера". 
                   Для векторов это может быть длина, а для матриц — различные нормы, например, 
                   Frobenius-норма или максимальная норма. Норма используется в различных областях, 
                   включая оптимизацию и анализ данных.""",
                   doc="numpy.linalg.norm",
                   example=example)

# numpy.linalg.svd
if menu == "numpy.linalg.svd":
    def example():
        code = st.text_area("Введите код:",
                            value="A = np.array([[3, 2], [3, 4]])\nU, S, Vt = np.linalg.svd(A)", height=100)
        if st.button("Вычислить"):
            U, S, Vt, A = exec_and_return(code, ["U", "S", "Vt", "A"])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write("Матрица A:")
                st.write(A)
            with col2:
                st.write("U:")
                st.write(U)
            with col3:
                st.write("S:")
                st.write(S)
            with col4:
                st.write("Vt:")
                st.write(Vt)
    
    define_function("numpy.linalg.svd",
                   """Выполняет сингулярное разложение матрицы, которое позволяет разложить 
                   матрицу A на три матрицы: U, S и Vt. Это разложение полезно для анализа структуры 
                   данных, уменьшения размерности и решения задач наименьших квадратов.""",
                   doc="numpy.linalg.svd",
                   example=example)

# numpy.linalg.cholesky
if menu == "numpy.linalg.cholesky":
    def example():
        code = st.text_area("Введите:",
                            value="A = np.array([[3, 2], [3, 4]])\nL = np.linalg.cholesky(A)", height=100)
        if st.button("Вычислить"):
            L, A = exec_and_return(code, ["L", "A"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Матрица A:")
                st.write(A)
            with col2:
                if A.shape[0] == A.shape[1]:
                    st.write("Разложение Холецкого (L):")
                    st.write(L)
                else:
                    st.write("Разложение Холецкого можно выполнить только для квадратной положительно определенной матрицы.")
    
    define_function("numpy.linalg.cholesky",
                   """Выполняет разложение Холецкого для положительно определенной матрицы. 
                   Это разложение позволяет представить матрицу A в виде произведения матрицы L и её 
                   транспонированной матрицы L^T, что упрощает решение систем линейных уравнений.""",
                   doc="numpy.linalg.cholesky",
                   example=example)

# numpy.linalg.pinv
if menu == "numpy.linalg.pinv":
    def example():
        code = st.text_area("Введите:",
                            value="A = np.array([[3, 2], [3, 4]])\npinv_A = np.linalg.pinv(A)", height=100)
        if st.button("Вычислить"):
            pinv_A, A = exec_and_return(code, ["pinv_A", "A"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Матрица A:")
                st.write(A)
            with col2:
                st.write("Псевдообратная матрица A:")
                st.write(pinv_A)
    
    define_function("numpy.linalg.pinv",
                   """Вычисляет псевдообратную матрицу с помощью сингулярного разложения матрицы. Псевдообратная матрица используется для решения систем линейных уравнений, 
                   когда матрица не является квадратной или вырожденной. Она минимизирует норму разности 
                   между вектором b и Ax, где A — матрица коэффициентов, а x — вектор переменных.""",
                   doc="numpy.linalg.pinv",
                   example=example)

# numpy.linalg.qr
if menu == "numpy.linalg.qr":
    def example():
        code = st.text_area("Введите:",
                            value="A = np.array([[3, 2], [3, 4]])\nQ, R = np.linalg.qr(A)", height=100)
        if st.button("Вычислить"):
            Q, R, A = exec_and_return(code, ["Q", "R", "A"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Матрица A:")
                st.write(A)
            with col2:
                st.write("Матрица Q:")
                st.write(Q)
            with col3:
                st.write("Матрица R:")
                st.write(R)
    
    define_function("numpy.linalg.qr",
                   """Выполняет QR-разложение матрицы, которое позволяет представить 
                   матрицу A в виде произведения матрицы Q (ортогональной) и матрицы R (верхней треугольной). 
                   Это разложение полезно для решения систем линейных уравнений и вычисления собственных значений.""",
                   doc="numpy.linalg.qr",
                   example=example,
                   code="numpy.linalg.qr(a, mode='reduced')")

# numpy.linalg.eigh
if menu == "numpy.linalg.eigh":
    def example():
        code = st.text_area("Введите:",
                            value="""symmetric_matrix = np.array([[1, 2], [2, 1]])\neigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)""", height=100)
        if st.button("Вычислить"):
            eigenvalues, eigenvectors, symmetric_matrix = exec_and_return(code, ["eigenvalues", "eigenvectors", "symmetric_matrix"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Симметричная матрица:")
                st.write(symmetric_matrix)
            with col2:
                st.write("Собственные значения:")
                st.write(eigenvalues)
            with col3:
                st.write("Собственные векторы:")
                st.write(eigenvectors)
    
    define_function("numpy.linalg.eigh",
                   """Находит собственные значения и собственные векторы для эрмитовой (симметричной) 
                   матрицы. Это разложение полезно в различных областях, включая физику и статистику, 
                   так как оно позволяет анализировать свойства симметричных систем.""",
                   doc="numpy.linalg.eigh",
                   example=example,
                   code="numpy.linalg.eigh(a, UPLO='L')")

# numpy.linalg.lstsq
if menu == "numpy.linalg.lstsq":
    def example():
        code = st.text_area("Введите:",
                            value="A = np.array([[3, 2], [3, 4]])\nb = np.array([5, 11])\nx, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)", height=100)
        if st.button("Вычислить"):
            x, A, b = exec_and_return(code, ["x", "A", "b"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Матрица:")
                st.write(A)
            with col2:
                st.write("Вектор:")
                st.write(b)
            with col3:
                st.write("Решение задачи наименьших квадратов:")
                st.write(x)
    
    define_function("numpy.linalg.lstsq",
                   """Решает задачу наименьших квадратов для системы линейных уравнений, 
                   минимизируя сумму квадратов отклонений между наблюдаемыми и предсказанными значениями. 
                   Это полезно в регрессионном анализе и других статистических методах.""",
                   doc="numpy.linalg.lstsq",
                   example=example,
                   code="numpy.linalg.lstsq(a, b, rcond=None)"
                   )

# Работа с полиномами
if menu == "numpy.polynomial":
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
""", height=400)

        if st.button("Запустить"):
            p, values, sum_poly, product_poly, roots = exec_and_return(code, ["p", "values", "sum_poly", "product_poly", "roots"])

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
