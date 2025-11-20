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
                   r"""Находит обратную матрицу для квадратной матрицы.  
Обратная матрица $A$ обозначается как $A^{-1}$ и удовлетворяет уравнению $A \cdot A^{-1} = I$, 
где $I$ — единичная матрица. Если матрица не является невырожденной (определитель равен нулю), то обратная матрица не существует.""",
                   doc="numpy.linalg.inv",
                   example=example,
                   code="numpy.linalg.inv(a)")

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
характеризующим свойства матрицы. Для квадратной матрицы $A$ определитель обозначается как $det(A)$ или $|A|$.\n
Определитель используется для проверки, является ли матрица вырожденной 
($det(A) = 0$), и в решении систем линейных уравнений (например, по правилу Крамера).""",
                   doc="numpy.linalg.det",
                   example=example,
                   code="numpy.linalg.det(a)")

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
        
*Собственные значения* ($\\lambda$) — это скаляры, которые показывают, 
как матрица растягивает или сжимает пространство. \n
*Собственные векторы* ($\\vec{v}$) — это направления, которые остаются 
неизменными при применении матрицы.
Для матрицы $\\mathbf{A}$ и ненулевого вектора $\\vec{v}$:

$$
\\mathbf{A} \\vec{v} = \\lambda \\vec{v}
$$

Собственные значения находятся из уравнения:

$$
det(\\mathbf{A} - \\lambda \\mathbf{I}) = 0
$$

где $\\mathbf{I}$ — единичная матрица.""",
                   doc="numpy.linalg.eig",
                   example=example,
                   code="numpy.linalg.eig(a)")

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
                   """Функция решает систему линейных уравнений вида $Ax = b$, где:
    
- $A$ — матрица коэффициентов  
- $x$ — вектор переменных  
- $b$ — вектор свободных членов  

Использует метод Гаусса для нахождения решения, если оно существует.""",
                   doc="numpy.linalg.solve",
                   example=example,
                   code="numpy.linalg.solve(a, b)")

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
					r"""Вычисляет норму вектора или матрицы — меру их "размера". 
					Для векторов это может быть евклидова длина $\|\mathbf{x}\|_{\substack{2}}$, 
					а для матриц — различные нормы, например, 
					**норма Фробениуса** $\|A\|_{\substack{F}} = \sqrt{\sum\limits_{i,j} |a_{ij}|^2}$
					или спектральная норма. Нормы используются в оптимизации, 
					машинном обучении и анализе данных.""",
                   doc="numpy.linalg.norm",
                   example=example,
                   code="linalg.norm(x, ord=None, axis=None, keepdims=False)")

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
                   r"""
					**Сингулярное разложение матрицы имеет вид:**  
					$\mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^T$
					
					**где:**
					- $\mathbf{U}$ — унитарная матрица левых сингулярных векторов
					- $\mathbf{S}$ — диагональная матрица сингулярных значений  
					- $\mathbf{V}^T$ — транспонированная унитарная матрица правых сингулярных векторов

					Это разложение полезно для анализа структуры данных, уменьшения размерности и решения задач наименьших	 квадратов.
					""",
                   doc="numpy.linalg.svd",
                   example=example,
                   code="linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)")

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
					r"""
					Выполняет разложение Холецкого для положительно определённой матрицы.
    
					Это разложение позволяет представить матрицу $\mathbf{A}$ в виде произведения 
					нижней треугольной матрицы $\mathbf{L}$ и её транспонированной матрицы $\mathbf{L}^T$:
    
					$\mathbf{A} = \mathbf{L} \mathbf{L}^T$
    
					Такое представление упрощает решение систем линейных уравнений.
						""",
                   doc="numpy.linalg.cholesky",
                   example=example,
                   code="linalg.cholesky(a, /, *, upper=False)")

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
					r"""
					Вычисляет псевдообратную матрицу с помощью сингулярного разложения матрицы.
    
					Псевдообратная матрица $\mathbf{A}^+$ используется для решения систем линейных уравнений, 
					когда матрица $\mathbf{A}$ не является квадратной или вырожденной.
    
					Для сингулярного разложения $\mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^T$
					псевдообратная матрица вычисляется как:
    
					$$
					\mathbf{A}^+ = \mathbf{V} \mathbf{S}^+ \mathbf{U}^T
					$$
    
					где $\mathbf{S}^+$ - матрица, полученная транспонированием $\mathbf{S}$
					и заменой ненулевых сингулярных значений на обратные.
    
					Псевдообратная матрица минимизирует норму разности между вектором $\mathbf{b}$ и $\mathbf{A}\mathbf{x}$:
 
					$$
					\min \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2
					$$
    
					где $\mathbf{A}$ - матрица коэффициентов, а $\mathbf{x}$ - вектор переменных.
					""",
                   doc="numpy.linalg.pinv",
                   example=example,
                   code="linalg.pinv(a, rcond=None, hermitian=False, *, rtol=<no value>)")

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
                   r"""Задача наименьших квадратов для системы линейных уравнений заключается в нахождении вектора неизвестных, который минимизирует сумму квадратов отклонений (остатков) между левыми и правыми частями уравнений. Такая задача возникает, когда система имеет больше уравнений, чем неизвестных, и не имеет точного решения. Решение находится с помощью формулы $$\^x=(X^{*}X)^{-1}X^{*}Y$$, где $$X$$ — матрица коэффициентов, $$Y$$ — вектор правых частей, а $$X^{*}$$ — транспонированная матрица $$X$$. """,
                   doc="numpy.linalg.lstsq",
                   example=example,
                   code="numpy.linalg.lstsq(a, b, rcond=None)"
                   )
