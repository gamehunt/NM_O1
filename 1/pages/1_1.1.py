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

def build_matrix(n: int) -> np.ndarray:
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
    return A

def build_f(n: int) -> np.ndarray:
    return np.ones(n, dtype=float)

def lu_decomposition(A: np.ndarray):
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
            Ukk = U[k, k]
            if abs(Ukk) < 1e-12:
                raise ZeroDivisionError("Нулевой диагональный элемент — нужна перестановка строк.")
            L[i, k] = (A[i, k] - s) / Ukk

    return L, U

def solve_lu(L: np.ndarray, U: np.ndarray, b: np.ndarray):
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        s = sum(L[i, j] * y[j] for j in range(i))
        y[i] = b[i] - s

    for i in reversed(range(n)):
        s = sum(U[i, j] * x[j] for j in range(i + 1, n))
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
        "Собственная реализация",
        "Решение через SciPy",
        "Сравнение методов",
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
start_time = time.perf_counter()
lu, piv = lu_factor(A)          # LU-разложение с выбором главного элемента
x_solution = lu_solve((lu, piv), f)  # Решение системы
computation_time = time.perf_counter() - start_time

print(f"Время выполнения: {computation_time:.6f} сек")
print(f"Решение: {x_solution}")''', language='python')

    st.subheader("Результаты решения")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Решение системы**")
        st.write("Вектор решения x:")

        # Красивое отображение вектора решения
        if n <= 10:
            for i, val in enumerate(x_scipy):
                st.write(f"x[{i}] = {val:.8f}")
        else:
            st.dataframe(pd.DataFrame({
                'Индекс': range(n),
                'Значение': x_scipy
            }).set_index('Индекс'), use_container_width=True)

    with col2:
        st.write("**Метрики производительности**")
        st.metric("Время выполнения", f"{t_scipy:.6f} сек")

        residual = np.linalg.norm(A @ x_scipy - f)
        st.metric("Норма невязки", f"{residual:.2e}")

        condition_number = np.linalg.cond(A)
        st.metric("Число обусловленности", f"{condition_number:.2e}")

    # Информация о перестановках
    st.subheader("Информация о перестановках строк")

    st.markdown("""
    **Вектор перестановок `piv`** показывает, какие строки матрицы были переставлены 
    в процессе LU-разложения с выбором главного элемента:
    """)

    piv_df = pd.DataFrame({
        'Новая позиция': range(len(piv)),
        'Исходная строка': piv
    })
    st.dataframe(piv_df, use_container_width=True)

    # Проверка точности решения
    st.subheader("Проверка точности решения")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Правая часть (исходная)**")
        st.write(f)

    with col2:
        st.write("**Правая часть (вычисленная)**")
        computed_f = A @ x_scipy
        st.write(computed_f)

    # Анализ ошибки
    st.write("**Анализ ошибки:**")

    error = np.abs(f - computed_f)
    max_error = np.max(error)
    mean_error = np.mean(error)

    error_col1, error_col2, error_col3 = st.columns(3)

    with error_col1:
        st.metric("Максимальная ошибка", f"{max_error:.2e}")

    with error_col2:
        st.metric("Средняя ошибка", f"{mean_error:.2e}")

    with error_col3:
        st.metric("Норма невязки", f"{residual:.2e}")

    # Визуализация структуры LU-разложения
    st.subheader("Структура LU-разложения")

    st.markdown("""
    SciPy возвращает объединенную матрицу, содержащую как L, так и U:
    - Верхняя треугольная часть + диагональ = матрица U
    - Нижняя треугольная часть (без диагонали) = матрица L (диагональ L = 1)
    """)

    # Показываем объединенную матрицу
    st.write("Объединенная матрица LU (первые 10x10 элементов):")
    display_size = min(10, n)
    lu_display = lu[:display_size, :display_size]
    st.dataframe(lu_display, use_container_width=True)

    # Преимущества SciPy
    st.subheader("Преимущества использования SciPy")

    advantages = [
        "**Высокая производительность** - оптимизированные реализации на C/Fortran",
        "**Численная устойчивость** - автоматический выбор главного элемента",
        "**Надежность** - обработка особых случаев и ошибок",
        "**Стандартизация** - общепринятый интерфейс и поведение",
        "**Дополнительные функции** - широкий набор математических инструментов"
    ]

    for advantage in advantages:
        st.markdown(f"- {advantage}")

    # Сравнение с собственным методом (если доступно)
    try:
        st.subheader("Сравнение с собственной реализацией")

        # Вычисление собственным методом
        t0_own = time.perf_counter()
        L_own, U_own = lu_decomposition(A)
        x_own = solve_lu(L_own, U_own, f)
        t_own = time.perf_counter() - t0_own

        comparison_col1, comparison_col2, comparison_col3 = st.columns(3)

        with comparison_col1:
            st.metric("Время SciPy", f"{t_scipy:.6f} сек")

        with comparison_col2:
            st.metric("Время собств.", f"{t_own:.6f} сек")

        with comparison_col3:
            speedup = t_own / t_scipy if t_scipy > 0 else 0
            st.metric("Ускорение", f"{speedup:.1f}x")

        if speedup > 1:
            st.success(f"SciPy работает в {speedup:.1f} раз быстрее собственной реализации")
        else:
            st.info("Производительность методов сопоставима")

    except Exception as e:
        st.info("Для полного сравнения перейдите в раздел 'Сравнение методов'")

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

    st.subheader("Матрица A")
    st.dataframe(A, use_container_width=True)

    st.subheader("Функция построения вектора правой части")

    st.code('''def build_f(n: int) -> np.ndarray:
    return np.ones(n, dtype=float)''', language='python')

    st.subheader("Вектор правой части f")
    st.write(f)

elif selected_section == "Теория LU-разложения":
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
    """)

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

    st.markdown("""
    **Решение системы после разложения:**

    1. **Прямой ход:** Решение $L y = f$
    2. **Обратный ход:** Решение $U x = y$
    """)

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

elif selected_section == "Собственная реализация":
    st.header("Собственная реализация LU-разложения")

    st.subheader("Код решения с помощью собственной реализации")

    st.code('''import numpy as np
import time

# Построение матрицы и правой части
A = build_matrix(n)
f = build_f(n)

# LU-разложение и решение системы
start_time = time.perf_counter()
L, U = lu_decomposition(A) # LU-разложение
x_solution = solve_lu(L, U, f) # Решение системы
computation_time = time.perf_counter() - start_time

print(f"Время выполнения: {computation_time:.6f} сек")
print(f"Решение: {x_solution}")''', language='python')

    try:
        # Вычисления
        t0 = time.perf_counter()
        L, U = lu_decomposition(A)
        x_own = solve_lu(L, U, f)
        t_own = time.perf_counter() - t0

        st.subheader("Результаты решения")

        # Результаты
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Матрица L")
            st.dataframe(L, use_container_width=True)

        with col2:
            st.subheader("Матрица U")
            st.dataframe(U, use_container_width=True)

        with col3:
            st.subheader("Решение x")
            st.write(x_own)

        # Метрики производительности
        st.subheader("Метрики производительности")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Время выполнения", f"{t_own:.6f} сек")

        with col2:
            residual = np.linalg.norm(A @ x_own - f)
            st.metric("Норма невязки", f"{residual:.2e}")

        # Проверка разложения
        st.subheader("Проверка корректности разложения")

        A_reconstructed = L @ U
        decomposition_error = np.linalg.norm(A - A_reconstructed)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Ошибка разложения", f"{decomposition_error:.2e}")

        with col2:
            if decomposition_error < 1e-10:
                st.success("Разложение корректно")
            else:
                st.error("Обнаружена ошибка в разложении")

    except ZeroDivisionError as e:
        st.error(f"Ошибка при вычислениях: {str(e)}")
        st.info("Рекомендуется использовать метод с выбором главного элемента")

elif selected_section == "Сравнение методов":
    st.header("Сравнение методов")

    # Вычисления для обоих методов
    try:
        t0_own = time.perf_counter()
        L, U = lu_decomposition(A)
        x_own = solve_lu(L, U, f)
        t_own = time.perf_counter() - t0_own

        t0_scipy = time.perf_counter()
        lu, piv = lu_factor(A)
        x_scipy = lu_solve((lu, piv), f)
        t_scipy = time.perf_counter() - t0_scipy

        # Метрики сравнения
        diff = np.linalg.norm(x_own - x_scipy)
        res_own = np.linalg.norm(A @ x_own - f)
        res_scipy = np.linalg.norm(A @ x_scipy - f)

        # Визуализация сравнения
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Собственная реализация")
            st.metric("Время", f"{t_own:.6f} сек")
            st.metric("Невязка", f"{res_own:.2e}")

        with col2:
            st.subheader("SciPy")
            st.metric("Время", f"{t_scipy:.6f} сек")
            st.metric("Невязка", f"{res_scipy:.2e}")

        with col3:
            st.subheader("Сравнение")
            st.metric("Разность решений", f"{diff:.2e}")
            speedup = t_own / t_scipy if t_scipy > 0 else 0
            st.metric("Ускорение SciPy", f"{speedup:.1f}x")

        # Анализ результатов
        st.subheader("Анализ результатов")

        if diff < 1e-12:
            st.success("Решения полностью совпадают")
        elif diff < 1e-8:
            st.info("Решения практически идентичны (малая погрешность)")
        else:
            st.warning("Обнаружена значительная разница между решениями")

        if res_own < 1e-12 and res_scipy < 1e-12:
            st.success("Оба метода дали точное решение")
        elif res_scipy < res_own:
            st.info("SciPy демонстрирует лучшую точность")
        else:
            st.info("Собственная реализация показывает хорошую точность")

    except Exception as e:
        st.error(f"Ошибка при сравнении методов: {str(e)}")

elif selected_section == "Эксперименты":
    st.header("Эксперименты с различными размерностями")

    st.markdown("""
    Исследование зависимости времени выполнения от размерности матрицы.
    """)

    max_n = st.slider("Максимальная размерность для эксперимента:", 10, 200, 100, step=10)
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
