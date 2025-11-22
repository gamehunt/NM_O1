import numpy as np
import pandas as pd
import streamlit as st
from scipy.linalg import lu_factor, lu_solve
import time

# -------------------------------------------
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# -------------------------------------------

st.set_page_config(
    page_title="LU-разложение | Научные вычисления",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------
# ФУНКЦИИ
# -------------------------------------------
def array2latex(arr):
    """
    Конвертирует любой двумерный numpy-массив в LaTeX-матрицу.
    Поддерживает произвольный размер (m x n).
    """
    m, n = arr.shape
    latex = r"\begin{pmatrix}"

    for i in range(m):
        row_items = []
        for j in range(n):
            val = arr[i, j]
            # округляем красивые числа
            if abs(val - round(val)) < 1e-12:
                row_items.append(f"{int(round(val))}")
            else:
                row_str = f"{val:.6g}"  # короткое научное или обычное представление
                row_items.append(row_str)

        latex += " & ".join(row_items)
        if i < m - 1:
            latex += r" \\ "

    latex += r"\end{pmatrix}"

    return latex

def st_matrix_to_latex(matrix):
    """Конвертирует numpy matrix в LaTeX строку с целыми числами"""
    n = matrix.shape[0]
    latex = r"\begin{pmatrix}"
    for i in range(n):
        for j in range(n):
            # Всегда отображаем как целые числа
            value = matrix[i, j]
            if abs(value - round(value)) < 1e-10:
                latex += f"{int(round(value))}"
            else:
                latex += f"{int(value)}"  # если не целое, все равно показываем как целое
            if j < n - 1:
                latex += " & "
        if i < n - 1:
            latex += r" \\ "
    latex += r"\end{pmatrix}"
    return latex

def st_vector_to_latex(vector):
    """Конвертирует numpy vector в LaTeX строку с целыми числами"""
    n = len(vector)
    latex = r"\begin{pmatrix}"
    for i in range(n):
        # Всегда отображаем как целые числа
        value = vector[i]
        if abs(value - round(value)) < 1e-10:
            latex += f"{int(round(value))}"
        else:
            latex += f"{int(value)}"  # если не целое, все равно показываем как целое
        if i < n - 1:
            latex += r" \\ "
    latex += r"\end{pmatrix}"
    return latex

def build_matrix(n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 1.0
            elif i < j:
                A[i, j] = -1.0
            elif i > j != n - 1:
                A[i, j] = 0
            elif j == n - 1:
                A[i, j] = 1.0
    return A

def build_f(n: int) -> np.ndarray:
    return np.ones(n, dtype=float)

def lu_decomposition(A: np.ndarray):
    """
    Выполняет LU-разложение матрицы A: A = L * U

    Где:
        L — нижняя треугольная матрица с единицами на диагонали
        U — верхняя треугольная матрица

    Аналитическая идея LU-разложения:
    ---------------------------------
    Матрица A представляется как произведение:

        A = L * U

    В разложении Дулитла (Doolittle):
        L имеет единицы на диагонали: L_ii = 1
        U — обычная верхняя треугольная

    Элементы матриц вычисляются по формулам:

    1. Формулы для U (k-я строка U):
       ----------------------------------
       Для j = k..n−1:
       U[k, j] = A[k, j] − Σ_{m=0..k−1} L[k, m] * U[m, j]

    2. Формулы для L (k-й столбец L):
       ----------------------------------
       Для i = k+1..n−1:
       L[i, k] = (A[i, k] − Σ_{m=0..k−1} L[i, m] * U[m, k]) / U[k, k]

    То есть на каждом шаге k:
      - вычисляем k-ю строку U
      - затем вычисляем k-й столбец L

    Разложение существует тогда, когда все U[k, k] ≠ 0.
    """

    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    U = np.zeros_like(A, dtype=float)

    # L_ii = 1 (разложение Дулитла)
    for i in range(n):
        L[i, i] = 1.0

    # Основной цикл по столбцам/строкам k
    for k in range(n):

        # ---------------------------
        # 1) Вычисляем k-ю строку U
        # ---------------------------
        # Формула:
        #   U[k, j] = A[k, j] − Σ L[k, m] * U[m, j]   (m = 0..k−1)
        for j in range(k, n):
            s = sum(L[k, m] * U[m, j] for m in range(k))
            U[k, j] = A[k, j] - s


        # ---------------------------
        # 2) Вычисляем k-й столбец L
        # ---------------------------
        # Формула:
        #   L[i, k] = (A[i, k] − Σ L[i, m] U[m, k]) / U[k, k]
        #   (i > k)
        for i in range(k + 1, n):
            s = sum(L[i, m] * U[m, k] for m in range(k))
            L[i, k] = (A[i, k] - s) / U[k, k]

    return L, U

def solve_lu(L: np.ndarray, U: np.ndarray, b: np.ndarray):
    """
    Решение системы линейных уравнений A x = b с использованием LU-разложения.
    Предполагается, что A = L * U.

    Идея аналитического решения:
    ----------------------------
    1. A x = b
       Подставляем A = L U:
           L U x = b

    2. Вводим новую переменную:
           U x = y

       Тогда получаем систему:
           L y = b

       Это система с нижней треугольной матрицей L.
       Решаем её прямым ходом.

    3. После нахождения y решаем:
           U x = y

       Это система с верхней треугольной матрицей U.
       Решаем её обратным ходом.
    """

    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    # -------------------------------
    # ПРЯМОЙ ХОД (решение L y = b)
    # -------------------------------
    # Нижняя треугольная система:
    #   y[i] = b[i] - sum(L[i, j] * y[j], j = 0..i-1)
    #
    # Так как L[i,i] = 1, делить не нужно.
    for i in range(n):
        # Вычисляем сумму Σ L_ij * y_j
        s = sum(L[i, j] * y[j] for j in range(i))
        # Получаем аналитическую формулу
        # y[i] = b[i] - Σ_{j<i} L_ij * y_j
        y[i] = b[i] - s

    # -------------------------------
    # ОБРАТНЫЙ ХОД (решение U x = y)
    # -------------------------------
    # Верхняя треугольная система:
    #   x[i] = (y[i] - sum(U[i, j] * x[j], j = i+1..n-1)) / U[i, i]
    #
    # Деление обязательно, потому что диагональ U != 1.
    for i in reversed(range(n)):
        # Вычисляем сумму Σ U_ij * x_j
        s = sum(U[i, j] * x[j] for j in range(i + 1, n))
        # Аналитическая формула:
        # x[i] = (y[i] - Σ_{j>i} U_ij * x_j) / U_ii
        x[i] = (y[i] - s) / U[i, i]

    return x

# -------------------------------------------
# ЗАГОЛОВОК И НАВИГАЦИЯ
# -------------------------------------------

st.title("LU-разложение матриц")

# Боковая панель с навигацией и параметрами
with st.sidebar:
    st.header("Навигация")
    sections = [
        "Постановка задачи",
        "Теория LU-разложения",
        "Применение",
        "Решение через SciPy",
        "Эксперименты"
    ]
    selected_section = st.radio("Перейти к разделу:", sections)

    st.header("Параметры вычислений")
    n = st.slider("Размерность матрицы n:", 2, 50, 8, step=1)

    st.markdown("---")
    st.markdown("""
    **LU-разложение** — метод решения систем линейных уравнений 
    путём разложения матрицы на нижнюю (L) и верхнюю (U) треугольные матрицы.
    """)

# Генерация данных
A = build_matrix(n)
f = build_f(n)

# -------------------------------------------
# РЕШЕНИЕ ЧЕРЕЗ SCIPY
# -------------------------------------------

if selected_section == "Решение через SciPy":
    st.header("Решение через SciPy")

    # Вычисления
    t0 = time.perf_counter()
    lu, piv = lu_factor(A)
    x_scipy = lu_solve((lu, piv), f)
    t_scipy = time.perf_counter() - t0

    st.subheader("Функции SciPy для LU-разложения")

    st.markdown("""
    **`scipy.linalg.lu_factor(A)`**
    - Выполняет LU-разложение с частичным выбором главного элемента
    - Возвращает объединенную матрицу LU и вектор перестановок piv
    - Автоматически обрабатывает нулевые элементы на диагонали
    
    **`scipy.linalg.lu_solve((lu, piv), b)`**
    - Решает систему линейных уравнений используя предвычисленное LU-разложение
    - Учитывает перестановки строк из вектора piv
    - Оптимизирована для высокой производительности
    """)

    st.subheader("Код решения через SciPy")

    st.code('''import numpy as np
from scipy.linalg import lu_factor, lu_solve
import time

# Построение матрицы и правой части
A = build_matrix(n)
f = build_f(n)

# LU-разложение и решение системы
lu, piv = lu_factor(A)          # LU-разложение с выбором главного элемента
x_solution = lu_solve((lu, piv), f)  # Решение системы''', language='python')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Решение системы")
        st.write("Вектор решения x:")

        # Красивое отображение вектора решения
        st.latex(r"\mathbf{x} = " + st_vector_to_latex(x_scipy))

    with col2:
        # Информация о перестановках
        st.subheader("Информация о перестановках строк")

        st.markdown("""
                **Вектор перестановок `piv`** показывает, какие строки матрицы были переставлены 
                в процессе LU-разложения с выбором главного элемента:
                """)

        st.latex(r"\mathbf{Исходная} = " + array2latex(np.array([range(len(piv)), piv]).T) + r" = \mathbf{piv}")

    # Визуализация структуры LU-разложения
    st.subheader("Структура LU-разложения")

    st.markdown("""
        SciPy возвращает объединенную матрицу, содержащую как L, так и U:
        - Верхняя треугольная часть + диагональ = матрица U
        - Нижняя треугольная часть (без диагонали) = матрица L (диагональ L = 1)
        """)

    # Показываем объединенную матрицу
    st.latex(r"\mathbf{LU} = " + st_matrix_to_latex(lu))

    # Преимущества SciPy
    st.subheader("Преимущества использования SciPy")

    advantages = [
        "**Высокая производительность** - оптимизированные реализации на C/Fortran",
        "**Численная устойчивость** - автоматический выбор главного элемента",
        "**Надежность** - обработка особых случаев и ошибок",
    ]

    for advantage in advantages:
        st.markdown(f"- {advantage}")

# -------------------------------------------
# ОСТАЛЬНЫЕ РАЗДЕЛЫ (кратко)
# -------------------------------------------

elif selected_section == "Постановка задачи":
    st.header("Постановка задачи")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Рассматривается система линейных алгебраических уравнений:

        $$
        A x = f
        $$

        **Матрица** $A=(a_{ij})$ задаётся по правилу:

        $$
        a_{ij} =
        \\begin{cases}
        1, & i = j,\\\\
        -1, & i < j,\\\\
        0, & i > j,\\; j \\ne n,\\\\
        1, & j = n,
        \\end{cases}
        \\qquad i,j = 1,\\dots,n,
        $$

        **Правая часть:**

        $$
        f_i = 1,\\quad i = 1,\\dots,n.
        $$
        """)

    with col2:
        st.info("**Текущие параметры**")
        st.write(f"Размерность: n = {n}")
        st.write(f"Размер матрицы: {n} × {n}")
        st.write(f"Размер вектора: {n}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Функция построения матрицы")

        st.code('''def build_matrix(n: int) -> np.ndarray:
            A = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        A[i, j] = 1.0
                    elif i < j:
                        A[i, j] = -1.0
                    elif i > j and j != n - 1:
                        A[i, j] = 0.0
                    elif j == n - 1:
                        A[i, j] = 1.0
            return A''', language='python')


    with col2:
        st.subheader("Матрица A")
        st.latex(r"\mathbf{A} = " + st_matrix_to_latex(A))


    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Функция построения вектора правой части")

        st.code('''def build_f(n: int) -> np.ndarray:
            return np.ones(n, dtype=float)''', language='python')

    with col2:
        st.subheader("Вектор правой части f")
        st.latex(r"\mathbf{f} = " + st_vector_to_latex(f))


elif selected_section == "Теория LU-разложения":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Теория LU-разложения")

        st.markdown("""
            LU-разложение — представление матрицы в виде:

            $$
            A = L U,
            $$

            где:

            - $L$ — нижняя треугольная матрица с единичной диагональю
            - $U$ — верхняя треугольная матрица

            **Алгоритм построения:**

            Для $k = 0, \\dots, n-1$:

            1. Вычисление элементов $U$:
               $$
               U_{kj} = a_{kj} - \\sum_{m=0}^{k-1} L_{km} U_{mj}, \\quad j = k, \\dots, n-1
               $$

            2. Вычисление элементов $L$:
               $$
               L_{ik} = \\frac{1}{U_{kk}} \\left( a_{ik} - \\sum_{m=0}^{k-1} L_{im} U_{mk} \\right), \\quad i = k+1, \\dots, n-1
               $$

            **Решение системы после разложения:**

            1. **Прямой ход:** Решение $L y = f$
            2. **Обратный ход:** Решение $U x = y$
            """)

    with col2:
        st.subheader("Реализация LU-разложения")

        st.code('''def lu_decomposition(A):
            n = A.shape[0]
            L = np.zeros_like(A)
            U = np.zeros_like(A)

            for i in range(n):
                L[i, i] = 1.0

            for k in range(n):
                for j in range(k, n):
                    s = sum(L[k, m] * U[m, j] for m in range(k))
                    U[k, j] = A[k, j] - s

                for i in range(k + 1, n):
                    s = sum(L[i, m] * U[m, k] for m in range(k))
                    L[i, k] = (A[i, k] - s) / U[k, k]

            return L, U''', language='python')

        st.subheader("Решение системы")

        st.code('''def solve_lu(L: np.ndarray, U: np.ndarray, b: np.ndarray):
        n = len(b)
        y = np.zeros(n)
        x = np.zeros(n)
    
        # Прямой ход: L y = b
        for i in range(n):
            s = sum(L[i, j] * y[j] for j in range(i))
            y[i] = b[i] - s
    
        # Обратный ход: U x = y
        for i in reversed(range(n)):
            s = sum(U[i, j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - s) / U[i, i]

        return x''', language='python')

elif selected_section == "Применение":
    st.header("Применение")

    st.code('''import numpy as np
import time

# Построение матрицы и правой части
A = build_matrix(n)
f = build_f(n)

# LU-разложение и решение системы
L, U = lu_decomposition(A) # LU-разложение
x_solution = solve_lu(L, U, f) # Решение системы''', language='python')

    try:
        L, U = lu_decomposition(A)
        x = solve_lu(L, U, f)

        st.subheader("Результаты решения")

        # Результаты
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Матрица L")
            st.latex(r"\mathbf{L} = " + st_matrix_to_latex(L))

        with col2:
            st.subheader("Матрица U")
            st.latex(r"\mathbf{U} = " + st_matrix_to_latex(U))

        with col3:
            st.subheader("Решение x")
            st.latex(r"\mathbf{x} = " + st_vector_to_latex(x))

        # Проверка разложения
        st.subheader("Проверка корректности разложения")

        "Функция `np.linalg.norm(v)` вычисляет норму вектора или матрицы."
        st.code('''A_reconstructed = L @ U # Матричное умножение
decomposition_error = np.linalg.norm(A - A_reconstructed)
if decomposition_error < 1e-10:
    st.success("Разложение корректно")
else:
    st.error("Обнаружена ошибка в разложении")''', language='python')
        A_reconstructed = L @ U
        decomposition_error = np.linalg.norm(A - A_reconstructed)

        if decomposition_error < 1e-10:
            st.success("Разложение корректно")
        else:
            st.error("Обнаружена ошибка в разложении")

    except ZeroDivisionError as e:
        st.error(f"Ошибка при вычислениях: {str(e)}")
        st.info("Рекомендуется использовать метод с выбором главного элемента")

elif selected_section == "Эксперименты":
    st.header("Эксперименты с различными размерностями")

    st.markdown("""
    Исследование зависимости времени выполнения от размерности матрицы.
    """)

    max_n = st.slider("Максимальная размерность для эксперимента:", 10, 400, 100, step=10)
    step = st.slider("Шаг изменения размерности:", 5, 20, 10, step=5)

    if st.button("Запустить эксперимент"):
        with st.spinner("Выполнение вычислений..."):
            ns = list(range(10, max_n + 1, step))
            times_own = []
            times_scipy = []

            progress_bar = st.progress(0)

            for idx, n_exp in enumerate(ns):
                A_exp = build_matrix(n_exp)
                f_exp = build_f(n_exp)

                # Собственный метод
                try:
                    t0 = time.perf_counter()
                    L_exp, U_exp = lu_decomposition(A_exp)
                    x_exp_own = solve_lu(L_exp, U_exp, f_exp)
                    t1 = time.perf_counter()
                    times_own.append(t1 - t0)
                except:
                    times_own.append(np.nan)

                # SciPy
                t0 = time.perf_counter()
                lu_exp, piv_exp = lu_factor(A_exp)
                x_exp_scipy = lu_solve((lu_exp, piv_exp), f_exp)
                t1 = time.perf_counter()
                times_scipy.append(t1 - t0)

                progress_bar.progress((idx + 1) / len(ns))

            # Создание DataFrame для визуализации
            df = pd.DataFrame({
                "n": ns,
                "Собственный метод": times_own,
                "SciPy": times_scipy
            }).set_index("n")

            st.subheader("График зависимости времени от размерности")
            st.line_chart(df, use_container_width=True)

            st.subheader("Таблица результатов")
            st.dataframe(df, use_container_width=True)

            # Анализ производительности
            st.subheader("Анализ производительности")

            avg_speedup = np.nanmean(np.array(times_own) / np.array(times_scipy))
            st.write(f"Среднее ускорение SciPy: **{avg_speedup:.1f}x**")

            if avg_speedup > 5:
                st.success("SciPy значительно превосходит собственную реализацию по скорости")
            elif avg_speedup > 2:
                st.info("SciPy демонстрирует заметное преимущество в производительности")
            else:
                st.info("Разница в производительности незначительна")
