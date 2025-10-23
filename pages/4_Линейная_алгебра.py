import streamlit as st
import numpy as np

# Заголовок приложения
st.title("Функции линейной алгебры в NumPy")
A = np.array([[3, 2], [3, 4]])
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
                            "Работа с полиномами",
                        )
                        )

if menu == "numpy.linalg.inv":
    f = 0
    st.header("numpy.linalg.inv")
    st.write("""Находит обратную матрицу для квадратной матрицы. 
    Обратная матрица A обозначается как A^(-1) и удовлетворяет уравнению A * A^(-1) = I, 
    где I — единичная матрица. Если матрица не является невырожденной (определитель равен нулю), 
    то обратная матрица не существует.""")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.linalg.inv.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для преобразования массива:",
                        value="A = np.array([[3, 2], [3, 4]])\ninv_A = np.linalg.inv(A)", height=100)
    if st.button("Вычислить обратную матрицу"):
        exec(code)
        f = 1
    col1, col2 = st.columns(2)
    with col1:
        st.write("Матрица A:")
        st.write(A)
    with col2:
        if f == 1:
            if A.shape[0] == A.shape[1]:  # Проверяем, является ли матрица квадратной
                st.write("Обратная матрица A:")
                st.write(inv_A)
            else:
                st.write("Матрица A не является квадратной, обратная матрица не существует.")

if menu == "numpy.linalg.det":
    st.header("numpy.linalg.det")
    st.write("""Вычисляет определитель матрицы, который является скалярным значением, 
    характеризующим свойства матрицы. Определитель используется для проверки, является ли 
    матрица вырожденной (определитель равен нулю) и в решении систем линейных уравнений.""")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.linalg.det.html"
    st.markdown(f"[Подробнее]({url})")


    code = st.text_area("Введите код для преобразования массива:",
        value="A = np.array([[3, 2], [3, 4]])\ndet_A = np.linalg.det(A)", height=100)
    if st.button("Вычислить определитель"):
        exec(code)


    col1, col2 = st.columns(2)

    with col1:
        st.write("Матрица A:")
        st.write(A)

    with col2:
        if A.shape[0] == A.shape[1]:  # Проверяем, является ли матрица квадратной
            st.write("Определитель матрицы A:")
            st.write(det_A)
        else:
            st.write("Определитель можно вычислить только для квадратной матрицы.")

if menu == "numpy.linalg.eig":
    st.header("numpy.linalg.eig")
    st.write("""Находит собственные значения и собственные векторы матрицы. 
    Собственные значения — это скаляры, которые показывают, как матрица растягивает 
    или сжимает пространство, а собственные векторы — это направления, которые остаются 
    неизменными при применении матрицы.""")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html"
    st.markdown(f"[Подробнее]({url})")


    code = st.text_area("Введите код для преобразования массива:",
                        value="A = np.array([[3, 2], [3, 4]])\neigenvalues, eigenvectors = np.linalg.eig(A)", height=100)
    if st.button("Вычислить собственные вектора"):
        exec(code)
    quad = 0
    col1, col2, col3 = st.columns(3)

    with col1:

        st.write("Матрица A:")
        st.write(A)
        if A.shape[0] == A.shape[1]:  # Провер яем, является ли матрица квадратной
            quad = 0
        else:
            quad = 1

    with col2:
        if quad == 0:
            st.write("Собственные значения:")
            st.write(eigenvalues)
        else: st.write("Собственные значения и векторы можно вычислить только для квадратной матрицы.")

    with col3:
        if quad == 0:
            st.write("Собственные векторы:")
            st.write(eigenvectors)
        else: st.write(" ")





if menu == "numpy.linalg.solve":
    st.header("numpy.linalg.solve")
    st.write("""Решает систему линейных уравнений вида Ax = b, где A — матрица коэффициентов, 
    x — вектор переменных, а b — вектор свободных членов. Использует метод Гаусса для нахождения 
    решения, если оно существует.""")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.linalg.solve.html"
    st.markdown(f"[Подробнее]({url})")


    code = st.text_area("Введите код для преобразования массива:",
                        value="A = np.array([[3, 2], [3, 4]])\nb = np.array([5, 11])\nx = np.linalg.solve(A, b)", height=100)
    if st.button("Вычислить СЛАУ"):
        exec(code)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Матрица A:")
        st.write(A)

    with col2:
        st.write("Вектор b:")
        st.write(b)
    with col3:

        if A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0]:  # Проверяем, что матрица квадратная и размерности совпадают

                st.write("Решение системы:")
                st.write(x)
        else:
            st.write("Система уравнений не может быть решена, так как размеры матрицы A и вектора b не совпадают.")

if menu == "numpy.linalg.norm":
    st.header("numpy.linalg.norm")
    st.write("""Вычисляет норму вектора или матрицы, которая является мерой их "размера". 
    Для векторов это может быть длина, а для матриц — различные нормы, например, 
    Frobenius-норма или максимальная норма. Норма используется в различных областях, 
    включая оптимизацию и анализ данных.""")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.linalg.norm.html"
    st.markdown(f"[Подробнее]({url})")


    code = st.text_area("Введите код для преобразования массива:",
                        value="A = np.array([[3, 2], [3, 4]])\nnorm_A = np.linalg.norm(A)", height=100)
    if st.button("Вычислить СЛАУ"):
        exec(code)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Матрица A:")
            st.write(A)
        with col2:
            st.write("Норма матрицы A:")
            st.write(norm_A)

if menu == "numpy.linalg.svd":
    st.header("numpy.linalg.svd")
    st.write("""Выполняет сингулярное разложение матрицы, которое позволяет разложить 
    матрицу A на три матрицы: U, S и Vt. Это разложение полезно для анализа структуры 
    данных, уменьшения размерности и решения задач наименьших квадратов.""")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.linalg.svd.html"
    st.markdown(f"[Подробнее]({url})")


    code = st.text_area("Введите код:",
                        value="A = np.array([[3, 2], [3, 4]])\nU, S, Vt = np.linalg.svd(A)", height=100)
    if st.button("Вычислить"):
        exec(code)
        U, S, Vt = np.linalg.svd(A)
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

if menu == "numpy.linalg.cholesky":
    st.header("numpy.linalg.cholesky")
    st.markdown("""
    Выполняет разложение Холецкого для положительно определенной матрицы. 
    Это разложение позволяет представить матрицу A в виде произведения матрицы L и её 
    транспонированной матрицы L^T, что упрощает решение систем линейных уравнений.
    """)

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.linalg.cholesky.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите:",
                        value="A = np.array([[3, 2], [3, 4]])\nL = np.linalg.cholesky(A)", height=100)

    if st.button("Вычислить"):
        exec(code)
    col1, col2 = st.columns(2)
    with col1:
        st.write("Матрица A:")
        st.write(A)

    with col2:
        if A.shape[0] == A.shape[1]:  # Проверяем, является ли матрица квадратной
            st.write("Разложение Холецкого (L):")
            st.write(L)
        else:
            st.write("Разложение Холецкого можно выполнить только для квадратной положительно определенной матрицы.")

if menu == "numpy.linalg.pinv":
    st.header("numpy.linalg.pinv")
    st.write("""Вычисляет псевдообратную матрицу с помощью сингулярного разложения матрицы. Псевдообратная матрица используется для решения систем линейных уравнений, 
    когда матрица не является квадратной или вырожденной. Она минимизирует норму разности 
    между вектором b и Ax, где A — матрица коэффициентов, а x — вектор переменных.""")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.linalg.pinv.html"
    st.markdown(f"[Подробнее]({url})")


    code = st.text_area("Введите:",
                        value="A = np.array([[3, 2], [3, 4]])\npinv_A = np.linalg.pinv(A)", height=100)
    if st.button("Вычислить"):
        exec(code)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Матрица A:")
            st.write(A)
        with col2:
            st.write("Псевдообратная матрица A:")
            st.write(pinv_A)

if menu == "numpy.linalg.qr":
    st.header("numpy.linalg.qr")
    st.write("""Выполняет QR-разложение матрицы, которое позволяет представить 
    матрицу A в виде произведения матрицы Q (ортогональной) и матрицы R (верхней треугольной). 
    Это разложение полезно для решения систем линейных уравнений и вычисления собственных значений.""")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.linalg.qr.html"
    st.markdown(f"[Подробнее]({url})")


    code = st.text_area("Введите:",
                        value="A = np.array([[3, 2], [3, 4]])\nQ, R = np.linalg.qr(A)", height=100)
    if st.button("Вычислить"):
        exec(code)

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

if menu == "numpy.linalg.eigh":
    st.header("numpy.linalg.eigh")
    st.write("""Находит собственные значения и собственные векторы для эрмитовой (симметричной) 
    матрицы. Это разложение полезно в различных областях, включая физику и статистику, 
    так как оно позволяет анализировать свойства симметричных систем.""")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eigh.html"
    st.markdown(f"[Подробнее]({url})")


    code = st.text_area("Введите:",
                        value="""A = np.array([[3, 2], [3, 4]])\nsymmetric_matrix = np.array([[1, 2], [2, 1]])\neigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)""", height=100)
    if st.button("Вычислить"):
        exec(code)

          # Пример симметричной матрицы

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

if menu == "numpy.linalg.lstsq":
    st.header("numpy.linalg.lstsq")
    st.write("""Решает задачу наименьших квадратов для системы линейных уравнений, 
    минимизируя сумму квадратов отклонений между наблюдаемыми и предсказанными значениями. 
    Это полезно в регрессионном анализе и других статистических методах.""")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.linalg.lstsq.html"
    st.markdown(f"[Подробнее]({url})")


    code = st.text_area("Введите:",
                        value="A = np.array([[3, 2], [3, 4]])\nb = np.array([5, 11])\nx, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)", height=100)
    if st.button("Вычислить"):
        exec(code)

      # Пример вектора свободных членов

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

if menu == "Работа с полиномами":
    st.header("Работа с полиномами в NumPy")
    st.write("Работа с полиномами в NumPy осуществляется с помощью модуля :red[numpy.polynomial]")

    url = "https://numpy.org/doc/stable/reference/generated/numpy.poly.html"
    st.markdown(f"[Подробнее]({url})")


    code = st.text_area("Интерактивный код:",
value="""# Создать полином:
coef = [2, 3, 1]  
p = np.poly1d(coef)

#Значения полинома p в точках x:
x = np.array([0, 1, 2])
values = p(x)

#Сумма двух полиномов:
q = np.poly1d([1, 4])
sum_poly = p + q

# Произведение полиномов p и q:
product_poly = p * q

# Корни полинома p:
roots = np.roots(coef)

""", height=500)

    if st.button("Запустить"):
        try:
            exec(code)
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
        except Exception as e:
            st.error(f"Ошибка: {e}")