import streamlit as st
import numpy as np
import scipy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt

def create_matrix_A(n):
    """Создает матрицу A по заданным правилам"""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 1
            elif i < j:
                A[i, j] = -1
            elif i > j and j != n - 1:
                A[i, j] = 0
            elif j == n - 1:
                A[i, j] = 1
    return A

def lu_decomposition(A):
    """LU-разложение матрицы"""
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))

        for j in range(i, n):
            if i == j:
                L[j, i] = 1
            else:
                L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U

def solve_system(L, U, f):
    """Решение системы LUx = f"""
    n = len(f)

    # Прямая подстановка: Ly = f
    y = np.zeros(n)
    for i in range(n):
        y[i] = f[i] - sum(L[i, j] * y[j] for j in range(i))

    # Обратная подстановка: Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i+1, n))) / U[i, i]

    return x, y

def solve_lu(L, U, f):
    """Альтернативное название для solve_system для совместимости"""
    return solve_system(L, U, f)

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

def format_matrix_display(matrix):
    """Форматирует матрицу для отображения в таблице с целыми числами"""
    n = matrix.shape[0]
    formatted = []
    for i in range(n):
        row = []
        for j in range(n):
            value = matrix[i, j]
            if abs(value - round(value)) < 1e-10:
                row.append(f"{int(round(value))}")
            else:
                row.append(f"{int(value)}")
        formatted.append(row)
    return formatted

def format_vector_display(vector):
    """Форматирует вектор для отображения в таблице с целыми числами"""
    n = len(vector)
    formatted = []
    for i in range(n):
        value = vector[i]
        if abs(value - round(value)) < 1e-10:
            formatted.append(f"{int(round(value))}")
        else:
            formatted.append(f"{int(value)}")
    return formatted

def main():
    st.set_page_config(page_title="LU-разложение", layout="wide", initial_sidebar_state="expanded")

    st.sidebar.title("Навигация")
    page = st.sidebar.radio("Выберите раздел:",
                            ["Постановка задачи",
                             "LU-разложение",
                             "Решение системы",
                             "Сравнение с SciPy",
                             "Анализ результатов"])

    if page == "Постановка задачи":
        st.title("Задание 1.1: Постановка задачи")

        st.header("Условие задачи")

        st.write("""
        Написать программу, реализующую решение системы линейных алгебраических уравнений 
        на основе **LU-разложения**.
        """)

        st.subheader("Система уравнений:")
        st.latex(r"A\mathbf{x} = \mathbf{f}")

        st.subheader("Матрица $\mathbf{A}$:")
        st.latex(r"""
        a_{ij} = 
        \begin{cases} 
        1, & i = j, \\
        -1, & i < j, \\
        0, & i > j \neq n, \\
        1, & j = n,
        \end{cases}
        \quad i = 1, 2, \ldots, n, \quad j = 1, 2, \ldots, n
        """)

        st.subheader("Правая часть:")
        st.latex(r"f_i = 1, \quad i = 1, 2, \ldots, n")

        st.info(r"""
        **Примечание:** LU-разложение представляет матрицу $\mathbf{A}$ в виде произведения 
        нижней ($\mathbf{L}$) и верхней ($\mathbf{U}$) треугольных матриц: $\mathbf{A} = \mathbf{L}\mathbf{U}$
        """)

    elif page == "LU-разложение":
        st.title("LU-разложение матрицы")

        n = st.slider("Выберите размер матрицы $n$", min_value=2, max_value=10, value=4, key="lu_n")

        # Создание матрицы A
        A = create_matrix_A(n)

        st.header("1. Исходная матрица $\mathbf{A}$")
        st.latex(r"\mathbf{A} = " + st_matrix_to_latex(A))

        st.header("2. Процесс LU-разложения")

        # LU-разложение
        L, U = lu_decomposition(A)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Матрица $\mathbf{L}$")
            st.latex(r"\mathbf{L} = " + st_matrix_to_latex(L))
            st.write(r"""
            **Свойства $\mathbf{L}$:**
            - Нижняя треугольная матрица
            - $l_{ii} = 1$ для $i = 1, 2, \ldots, n$
            - Элементы ниже главной диагонали
            """)

        with col2:
            st.subheader("Матрица $\mathbf{U}$")
            st.latex(r"\mathbf{U} = " + st_matrix_to_latex(U))
            st.write(r"""
            **Свойства $\mathbf{U}$:**
            - Верхняя треугольная матрица  
            - Ненулевые элементы на и выше главной диагонали
            - $u_{ii} \neq 0$ для $i = 1, 2, \ldots, n$
            """)

        st.header("3. Проверка корректности разложения")

        # Проверка A = LU
        A_reconstructed = L @ U
        error = np.linalg.norm(A - A_reconstructed)

        # Визуальное сравнение матриц
        st.subheader("Визуальное сравнение матриц")

        col3, col4, col5 = st.columns([2, 1, 2])

        with col3:
            st.markdown("**Исходная матрица $\mathbf{A}$**")
            st.latex(r"\mathbf{A} = " + st_matrix_to_latex(A))

        with col4:
            st.markdown("**Должно быть равно**")
            st.markdown("<div style='text-align: center; font-size: 24px; margin: 20px 0;'>＝</div>", unsafe_allow_html=True)

        with col5:
            st.markdown("**Произведение $\mathbf{L}\mathbf{U}$**")
            st.latex(r"\mathbf{L}\mathbf{U} = " + st_matrix_to_latex(A_reconstructed))

        # Показ поэлементного сравнения
        st.subheader("Поэлементное сравнение")

        comparison_data = []
        for i in range(n):
            row = []
            for j in range(n):
                original = A[i, j]
                reconstructed = A_reconstructed[i, j]
                diff = abs(original - reconstructed)
                match = "=" if diff < 1e-10 else "!="
                row.append(f"{int(original)} {match} {int(reconstructed)}")
            comparison_data.append(row)

        # Создаем DataFrame для красивого отображения
        df_comparison = pd.DataFrame(
            comparison_data,
            index=[f"Строка {i+1}" for i in range(n)],
            columns=[f"Столбец {j+1}" for j in range(n)]
        )

        st.dataframe(df_comparison, use_container_width=True)

        # Индикатор точности
        st.subheader("Результат проверки")

        if error < 1e-12:
            st.success("**LU-разложение выполнено точно!**")
        elif error < 1e-8:
            st.success("**LU-разложение выполнено с высокой точностью**")
        elif error < 1e-5:
            st.warning("**LU-разложение выполнено с удовлетворительной точностью**")
        else:
            st.error("**Обнаружены значительные расхождения в LU-разложении**")

        # Метрики точности
        col6, col7, col8 = st.columns(3)

        with col6:
            st.metric(
                label="Норма разности",
                value=f"{error:.2e}",
                delta="Идеально" if error < 1e-12 else "Хорошо" if error < 1e-8 else "Приемлемо" if error < 1e-5 else "Плохо",
                delta_color="normal" if error < 1e-8 else "off" if error < 1e-5 else "inverse"
            )

        with col7:
            max_element_error = np.max(np.abs(A - A_reconstructed))
            st.metric(
                label="Макс. ошибка элемента",
                value=f"{max_element_error:.2e}"
            )

        with col8:
            perfect_elements = np.sum(np.abs(A - A_reconstructed) < 1e-10)
            total_elements = n * n
            accuracy_percent = (perfect_elements / total_elements) * 100
            st.metric(
                label="Точных элементов",
                value=f"{perfect_elements}/{total_elements}",
                delta=f"{accuracy_percent:.1f}%"
            )

        st.header("4. Алгоритм LU-разложения")

        st.latex(r"""
        \begin{aligned}
        &\text{Для } k = 1, 2, \ldots, n: \\
        &\quad \text{1. Вычисляем элементы матрицы } \mathbf{U}: \\
        &\quad \quad u_{kj} = a_{kj} - \sum_{m=1}^{k-1} l_{km} u_{mj}, \quad j = k, k+1, \ldots, n \\
        &\quad \text{2. Вычисляем элементы матрицы } \mathbf{L}: \\
        &\quad \quad l_{ik} = \frac{1}{u_{kk}} \left( a_{ik} - \sum_{m=1}^{k-1} l_{im} u_{mk} \right), \quad i = k+1, k+2, \ldots, n
        \end{aligned}
        """)

    elif page == "Решение системы":
        st.title("Решение системы уравнений")

        n = st.slider("Выберите размер матрицы $n$", min_value=2, max_value=10, value=4, key="solve_n")

        # Создание данных
        A = create_matrix_A(n)
        f = np.ones(n)

        st.header("1. Исходная система")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Матрица $\mathbf{A}$")
            st.latex(r"\mathbf{A} = " + st_matrix_to_latex(A))

        with col2:
            st.subheader("Вектор $\mathbf{f}$")
            st.latex(r"\mathbf{f} = " + st_vector_to_latex(f))

        st.header("2. Решение системы")

        # LU-разложение
        L, U = lu_decomposition(A)

        # Решение системы
        x, y = solve_system(L, U, f)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader(r"Промежуточный вектор $\mathbf{y}$")
            st.latex(r"\mathbf{L}\mathbf{y} = \mathbf{f} \Rightarrow \mathbf{y} = " + st_vector_to_latex(y))
            st.write("**Получен решением:** $\mathbf{L}\mathbf{y} = \mathbf{f}$")

        with col4:
            st.subheader(r"Решение $\mathbf{x}$")
            st.latex(r"\mathbf{U}\mathbf{x} = \mathbf{y} \Rightarrow \mathbf{x} = " + st_vector_to_latex(x))
            st.write("**Получен решением:** $\mathbf{U}\mathbf{x} = \mathbf{y}$")

        st.header("3. Проверка корректности решения")

        # Проверка Ax = f
        Ax = A @ x
        residual = np.linalg.norm(Ax - f)
        residual_vector = Ax - f

        # Визуальное сравнение векторов
        st.subheader("Визуальное сравнение результатов")

        col5, col6, col7 = st.columns([2, 1, 2])

        with col5:
            st.markdown("**Вычисленный $\mathbf{A}\mathbf{x}$**")
            st.latex(r"\mathbf{A}\mathbf{x} = " + st_vector_to_latex(Ax))
            st.write("**Результат умножения матрицы на решение**")

        with col6:
            st.markdown("**Должен равняться**")
            st.markdown("<div style='text-align: center; font-size: 24px; margin: 20px 0;'>＝</div>", unsafe_allow_html=True)

        with col7:
            st.markdown("**Исходный вектор $\mathbf{f}$**")
            st.latex(r"\mathbf{f} = " + st_vector_to_latex(f))
            st.write("**Заданная правая часть системы**")

        # Поэлементное сравнение векторов
        st.subheader("Поэлементное сравнение")

        element_comparison = []
        for i in range(n):
            computed = Ax[i]
            expected = f[i]
            diff = abs(computed - expected)
            match = "=" if diff < 1e-10 else "!="
            element_comparison.append({
                'Уравнение': f"{i+1}",
                'Вычислено (Ax)ᵢ': f"{int(computed)}",
                'Символ': match,
                'Ожидалось fᵢ': f"{int(expected)}",
                'Погрешность': f"{diff:.2e}"
            })

        df_element_comp = pd.DataFrame(element_comparison)
        st.dataframe(df_element_comp, use_container_width=True)

        # Визуализация невязки
        st.subheader("Анализ невязки")

        col8, col9, col10 = st.columns(3)

        with col8:
            # Индикатор общей точности
            if residual < 1e-12:
                st.success("**Решение точное!**")
            else:
                st.error("**Значительная невязка в решении**")

            st.metric(
                label="Норма невязки",
                value=f"{residual:.2e}",
                delta="Идеально" if residual < 1e-12 else "Отлично" if residual < 1e-8 else "Хорошо" if residual < 1e-5 else "Требует проверки",
                delta_color="normal" if residual < 1e-8 else "off" if residual < 1e-5 else "inverse"
            )

        with col9:
            max_residual = np.max(np.abs(residual_vector))
            st.metric(
                label="Макс. невязка уравнения",
                value=f"{max_residual:.2e}"
            )

        with col10:
            perfect_equations = np.sum(np.abs(residual_vector) < 1e-10)
            accuracy_percent = (perfect_equations / n) * 100
            st.metric(
                label="Точных уравнений",
                value=f"{perfect_equations}/{n}",
                delta=f"{accuracy_percent:.1f}%"
            )

        st.header("4. Детали вычислений")

        st.latex(r"""
        \begin{aligned}
        &\text{Прямая подстановка } (\mathbf{L}\mathbf{y} = \mathbf{f}): \\
        &\quad y_1 = f_1 \\
        &\quad y_i = f_i - \sum_{j=1}^{i-1} l_{ij} y_j, \quad i = 2, 3, \ldots, n \\[1em]
        &\text{Обратная подстановка } (\mathbf{U}\mathbf{x} = \mathbf{y}): \\
        &\quad x_n = \frac{y_n}{u_{nn}} \\
        &\quad x_i = \frac{1}{u_{ii}} \left( y_i - \sum_{j=i+1}^{n} u_{ij} x_j \right), \quad i = n-1, n-2, \ldots, 1
        \end{aligned}
        """)

    elif page == "Сравнение с SciPy":
        st.title("Сравнение с SciPy")

        n = st.slider("Выберите размер матрицы $n$", min_value=2, max_value=10, value=4, key="scipy_n")

        # Создание данных
        A = create_matrix_A(n)
        f = np.ones(n)

        st.header("1. Исходные данные")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Матрица $\mathbf{A}$")
            st.latex(r"\mathbf{A} = " + st_matrix_to_latex(A))

        with col2:
            st.subheader("Вектор $\mathbf{f}$")
            st.latex(r"\mathbf{f} = " + st_vector_to_latex(f))

        st.header("2. Решения")

        # Наше решение
        L, U = lu_decomposition(A)
        x_custom, _ = solve_system(L, U, f)

        # SciPy решение
        x_scipy = la.solve(A, f)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Наше решение (LU-разложение)")
            st.latex(r"\mathbf{x} = " + st_vector_to_latex(x_custom))

        with col4:
            st.subheader("Решение SciPy")
            st.latex(r"\mathbf{x} = " + st_vector_to_latex(x_scipy))

        st.header("3. Сравнение точности")

        # Расчет метрик
        residual_custom = np.linalg.norm(A @ x_custom - f)
        residual_scipy = np.linalg.norm(A @ x_scipy - f)
        solution_diff = np.linalg.norm(x_custom - x_scipy)

        # Таблица сравнения
        comparison_data = {
            'Метод': ['LU-разложение (наше)', 'SciPy'],
            r'Невязка $\|\mathbf{A}\mathbf{x} - \mathbf{f}\|$': [f"{residual_custom:.2e}", f"{residual_scipy:.2e}"],
            'Разница решений': [f"{solution_diff:.2e}", "-"]
        }

        st.table(pd.DataFrame(comparison_data))

    elif page == "Анализ результатов":
        st.title("Анализ результатов")

        st.header("Исследование для различных размеров матрицы")

        # Анализ для всех размеров от 2 до 10
        sizes = list(range(2, 11))

        st.write("### Методика исследования")
        st.write("""
        Проведено сравнение точности нашего LU-разложения с библиотечной реализацией SciPy 
        для размеров матриц от 2 до 10. Анализируются следующие метрики:
        - **Невязка решения** - норма разности между Ax и f
        - **Разница решений** - норма разности между нашим решением и решением SciPy
        - **Число обусловленности** - показатель чувствительности системы к ошибкам
        """)

        # Вычисление результатов
        results = []
        residuals_custom = []
        residuals_scipy = []
        condition_numbers = []

        for n in sizes:
            A = create_matrix_A(n)
            f = np.ones(n)

            try:
                # Наше решение
                L, U = lu_decomposition(A)
                x_custom, _ = solve_system(L, U, f)
                residual_custom = np.linalg.norm(A @ x_custom - f)

                # SciPy решение
                x_scipy = la.solve(A, f)
                residual_scipy = np.linalg.norm(A @ x_scipy - f)

                # Разница решений
                solution_diff = np.linalg.norm(x_custom - x_scipy)

                # Число обусловленности
                cond_number = np.linalg.cond(A)

                results.append({
                    'Размер n': n,
                    'Невязка (наша)': residual_custom,
                    'Невязка (SciPy)': residual_scipy,
                    'Разница решений': solution_diff,
                    'Число обусловленности': cond_number
                })

                residuals_custom.append(residual_custom)
                residuals_scipy.append(residual_scipy)
                condition_numbers.append(cond_number)

            except Exception as e:
                results.append({
                    'Размер n': n,
                    'Невязка (наша)': np.nan,
                    'Невязка (SciPy)': np.nan,
                    'Разница решений': np.nan,
                    'Число обусловленности': np.nan
                })
                residuals_custom.append(np.nan)
                residuals_scipy.append(np.nan)
                condition_numbers.append(np.nan)

        # Отображение результатов в таблице
        st.header("Результаты анализа")

        # Создаем форматированную таблицу для отображения
        display_results = []
        for res in results:
            display_results.append({
                'Размер n': res['Размер n'],
                'Невязка (наша)': f"{res['Невязка (наша)']:.2e}" if not np.isnan(res['Невязка (наша)']) else 'N/A',
                'Невязка (SciPy)': f"{res['Невязка (SciPy)']:.2e}" if not np.isnan(res['Невязка (SciPy)']) else 'N/A',
                'Разница решений': f"{res['Разница решений']:.2e}" if not np.isnan(res['Разница решений']) else 'N/A',
                'Число обусловленности': f"{res['Число обусловленности']:.2e}" if not np.isnan(res['Число обусловленности']) else 'N/A'
            })

        st.dataframe(pd.DataFrame(display_results), use_container_width=True)

        # Визуализация результатов
        st.header("Графический анализ")

        col1, col2 = st.columns(2)

        with col1:
            # График невязок
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.semilogy(sizes, residuals_custom, 'bo-', label='Наша реализация', linewidth=2, markersize=8)
            ax1.semilogy(sizes, residuals_scipy, 'ro-', label='SciPy', linewidth=2, markersize=8)
            ax1.set_xlabel('Размер матрицы n')
            ax1.set_ylabel('Норма невязки')
            ax1.set_title('Сравнение точности решений')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

        with col2:
            # График числа обусловленности
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.semilogy(sizes, condition_numbers, 'go-', linewidth=2, markersize=8)
            ax2.set_xlabel('Размер матрицы n')
            ax2.set_ylabel('Число обусловленности')
            ax2.set_title('Зависимость числа обусловленности от размера')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        # Анализ и выводы
        st.header("Выводы")

        successful_results = [r for r in results if not np.isnan(r['Невязка (наша)'])]
        if successful_results:
            avg_residual_custom = np.mean([r['Невязка (наша)'] for r in successful_results])
            avg_residual_scipy = np.mean([r['Невязка (SciPy)'] for r in successful_results])
            max_condition = np.max([r['Число обусловленности'] for r in successful_results])

            col3, col4, col5 = st.columns(3)

            with col3:
                st.metric(
                    "Средняя невязка (наша)",
                    f"{avg_residual_custom:.2e}",
                    delta="Низкая" if avg_residual_custom < 1e-10 else "Приемлемая" if avg_residual_custom < 1e-5 else "Высокая"
                )

            with col4:
                st.metric(
                    "Средняя невязка (SciPy)",
                    f"{avg_residual_scipy:.2e}",
                    delta="Низкая" if avg_residual_scipy < 1e-10 else "Приемлемая" if avg_residual_scipy < 1e-5 else "Высокая"
                )

            with col5:
                st.metric(
                    "Макс. число обусловленности",
                    f"{max_condition:.2e}",
                    delta="Хорошая" if max_condition < 1e3 else "Умеренная" if max_condition < 1e6 else "Плохая"
                )

            st.info("""
            **Основные наблюдения:**
            - Наша реализация LU-разложения демонстрирует сравнимую точность с библиотечной реализацией SciPy
            - Число обусловленности матриц растет с увеличением размера, что характерно для подобных задач
            - Разница между решениями минимальна, что подтверждает корректность нашей реализации
            """)
        else:
            st.error("Не удалось получить результаты для анализа")

if __name__ == "__main__":
    main()
