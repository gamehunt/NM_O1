import numpy as np
import scipy.linalg
import streamlit as st
import matplotlib.pyplot as plt

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i+1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L

def hilbert_matrix(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 / (i + j + 1)
    return H

def is_positive_definite(matrix):
    """Проверка матрицы на положительную определенность"""
    try:
        # Пытаемся выполнить разложение Холецкого
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def compute_condition_numbers(max_n):
    """Вычисление чисел обусловленности для матриц Гильберта разных размеров"""
    sizes = list(range(2, max_n + 1))
    condition_numbers = []
    
    for n in sizes:
        H = hilbert_matrix(n)
        cond_number = np.linalg.cond(H)
        condition_numbers.append(cond_number)
    
    return sizes, condition_numbers

def compute_leading_principal_minors(matrix):
    """Вычисление ведущих главных миноров матрицы"""
    n = matrix.shape[0]
    minors = []
    for i in range(1, n + 1):
        submatrix = matrix[:i, :i]
        minor = np.linalg.det(submatrix)
        minors.append(minor)
    return minors

# Создаем боковую панель для навигации с радиокнопками
st.sidebar.title("Навигация")

# Главный раздел с радиокнопками
main_section = st.sidebar.radio(
    "Выберите раздел:",
    ["Разложение Холецкого и определитель матрицы Гильберта", "Основные обозначения", "Постановка задачи", "Алгоритм разложения Холецкого", "Матрица Гильберта", "Реализация и сравнение"]
)

if main_section == "Разложение Холецкого и определитель матрицы Гильберта":
    st.title("Разложение Холецкого и определитель матрицы Гильберта")
    

elif main_section == "Основные обозначения":
    
    st.markdown("""
    ### Математические обозначения:
    - **$A$** - исходная симметричная положительно определённая матрица
    - **$L$** - нижняя треугольная матрица в разложении Холецкого
    - **$L^T$** - транспонированная матрица $L$
    - **$H$** - матрица Гильберта
    - **$n$** - размерность матрицы
    - **$a_{ij}$** - элемент матрицы $A$ в строке $i$, столбце $j$
    - **$l_{ij}$** - элемент матрицы $L$ в строке $i$, столбце $j$
    - **$\\det(A)$** - определитель матрицы $A$
    - **$\\kappa(A)$** - число обусловленности матрицы $A$
    
    ### Обозначения в коде:
    - **`A`** - входная матрица для разложения
    - **`L`** - результирующая нижняя треугольная матрица
    - **`H`** - матрица Гильберта
    - **`n`** - размер матрицы
    - **`i, j`** - индексы строк и столбцов
    - **`s`** - вспомогательная переменная для суммы произведений
    - **`cholesky_decomposition(A)`** - функция для вычисления разложения Холецкого
    - **`hilbert_matrix(n)`** - функция для генерации матрицы Гильберта
    - **`scipy.linalg.cholesky()`** - встроенная функция SciPy для разложения Холецкого
    - **`np.linalg.det()`** - встроенная функция NumPy для вычисления определителя
    - **`np.linalg.cond()`** - функция для вычисления числа обусловленности
    """)

elif main_section == "Постановка задачи":
    st.header("Постановка задачи")
    st.markdown("""
    Необходимо написать программу, реализующую разложение Холецкого $A = LL^T$ для симметричной положительно определённой матрицы $A$ и вычисляющую определитель матрицы на основе этого разложения.

    Найдите разложение Холецкого и определитель матрицы Гильберта, для которой

    $a_{ij} = \\frac{1}{i + j - 1}, \\quad i = 1, 2, \\dots, n, \\quad j = 1, 2, \\dots, n$

    при различных $n$.

    Решите также эту задачу с помощью библиотеки SciPy.
    """)

elif main_section == "Алгоритм разложения Холецкого":

    st.markdown(r"""
    ### Математические формулы разложения Холецкого

    Для симметричной положительно определённой матриции $A$ разложение $A = LL^T$ 
    вычисляется по следующим формулам:
    """)

    st.latex(r"""
    \begin{aligned}
    &l_{11} = \sqrt{a_{11}}, \\
    &l_{j1} = \frac{a_{j1}}{l_{11}}, \quad j = 2, \dots, n, \\
    &l_{ii} = \sqrt{a_{ii} - \sum_{p=1}^{i-1} l_{ip}^2}, \quad i = 2, \dots, n, \\
    &l_{ji} = \frac{1}{l_{ii}} \left( a_{ji} - \sum_{p=1}^{i-1} l_{ip} l_{jp} \right), \quad i = 2, \dots, n-1, \quad j = i+1, \dots, n.
    \end{aligned}
    """)

    st.markdown(r"""
    ### Вычислительный алгоритм
    """)

    st.markdown("**Шаг 1:** Инициализация")
    st.markdown("- Создаём нулевую матрицу $L$ того же размеры, что и $A$")

    st.markdown("**Шаг 2:** Вычисление элементов матрицы $L$")
    st.markdown("Для каждого $i = 0, 1, \\dots, n-1$ и $j = 0, 1, \\dots, i$:")

    st.markdown("1. **Вычисляем сумму скалярных произведений:**")
    st.latex(r"s = \sum_{k=0}^{j-1} L_{ik} \cdot L_{jk}")

    st.markdown("2. **Для диагональных элементов** ($i = j$):")
    st.latex(r"L_{ii} = \sqrt{A_{ii} - s}")

    st.markdown("3. **Для внедиагональных элементов** ($i > j$):")
    st.latex(r"L_{ij} = \frac{A_{ij} - s}{L_{jj}}")

    st.markdown("### Вычисление определителя")

    st.markdown("После получения разложения $A = LL^T$, определитель вычисляется как:")
    st.latex(r"\det(A) = \det(L) \cdot \det(L^T) = \det(L)^2 = \left( \prod_{i=0}^{n-1} L_{ii} \right)^2")

    st.markdown("### Особенности алгоритма:")
    st.markdown("- **Порядок вычислений**: сверху вниз, слева направо")
    st.markdown("- **Требования**: матрица должна быть симметричной и положительно определённой")
    st.markdown("- **Стабильность**: не требует выбора ведущего элемента")
    st.markdown("- **Эффективность**: сложность $O(n^3/3)$ операций")

elif main_section == "Матрица Гильберта":
    
    st.subheader("Особенности матрицы Гильберта")

    st.markdown(r"""
    Матрица Гильберта определяется элементами:
    $$H_{ij} = \frac{1}{i + j - 1}$$

    **Свойства:**
    - Симметричная и положительно определённая
    - Сильно обусловленная (число обусловленности растёт экспоненциально с n)
    - Определитель стремится к нулю при увеличении n
    - Используется как тестовый пример для численных алгоритмов

    **Примечание:** При больших n матрица Гильберта становится плохо обусловленной,
    что может приводить к вычислительным ошибкам.
    """)
    
    st.markdown("### Реализация матрицы Гильберта")
    st.code("""
def hilbert_matrix(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 / (i + j + 1)
    return H
    """, language='python')
    
    st.markdown("""
    **Пояснение реализации:**
    - Функция `hilbert_matrix(n)` создает матрицу размером `n × n`
    - Двойной цикл заполняет каждый элемент по формуле $H_{ij} = \\frac{1}{i + j + 1}$
    - Индексы `i` и `j` начинаются с 0, поэтому в знаменателе `i + j + 1` вместо `i + j - 1`
    """)
    
    n = st.slider("Выберите размер матрицы Гильберта n:", min_value=2, max_value=20, value=5, key="hilbert_slider")

    H = hilbert_matrix(n)
    
    st.subheader("Матрица Гильберта")
    st.write(H)
    
    # Проверка на положительную определенность
    st.subheader("Проверка на положительную определенность")
    
    # Проверка через собственные значения
    st.markdown("**Проверка через собственные значения:**")
    eigenvalues = np.linalg.eigvals(H)
    
    # Создаем красивый вывод для собственных значений
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Собственные значения:**")
        for i, val in enumerate(eigenvalues):
            st.write(f"$\lambda_{i+1}$ = {val:.10e}")
    
    with col2:
        min_eigenvalue = np.min(eigenvalues)
        max_eigenvalue = np.max(eigenvalues)
        all_positive = np.all(eigenvalues > 0)
        
        if all_positive:
            st.write("Все собственные значения положительны")
        else:
            st.write("Не все собственные значения положительны")
    
    # Проверка по критерию Сильвестра
    st.markdown("**Проверка по критерию Сильвестра (ведущие главные миноры):**")
    minors = compute_leading_principal_minors(H)
    
    # Создаем красивый вывод для миноров
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Главные миноры:**")
        for i, minor in enumerate(minors):
            st.write(f"$\Delta_{i+1}$: {minor:.10e}")
    
    with col2:
        all_minors_positive = all(minor > 0 for minor in minors)
        if all_minors_positive:
            st.write("Все ведущие главные миноры положительны")
        else:
            st.write("Не все ведущие главные миноры положительны")
    
    # Итоговый вывод
    st.markdown("**Итоговый вывод:**")
    if all_positive and all_minors_positive:
        st.markdown("""
        <div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; border-left: 5px solid #4CAF50;">
        <h4 style="color: #2e7d32; margin: 0;">Матрица положительно определена</h4>
        <p style="margin: 5px 0 0 0; color: #2e7d32;">Матрица Гильберта размером {n}×{n} является положительно определённой. 
        Разложение Холецкого возможно.</p>
        </div>
        """.format(n=n), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #ffebee; padding: 15px; border-radius: 5px; border-left: 5px solid #f44336;">
        <h4 style="color: #c62828; margin: 0;">Матрица не положительно определена</h4>
        <p style="margin: 5px 0 0 0; color: #c62828;">Матрица Гильберта размером {n}×{n} не является положительно определённой. 
        Разложение Холецкого невозможно.</p>
        </div>
        """.format(n=n), unsafe_allow_html=True)
    
    # Число обусловленности
    st.subheader("Число обусловленности")
    cond_number = np.linalg.cond(H)
    
    # Создаем стилизованный вывод для числа обусловленности
    if cond_number > 1e10:
        st.markdown(f"""
        <div style="background-color: #fff3e0; padding: 15px; border-radius: 5px; border-left: 5px solid #ff9800;">
        <h4 style="color: #e65100; margin: 0;">Очень плохая обусловленность</h4>
        <p style="margin: 5px 0 0 0; color: #e65100;">Число обусловленности: <strong>{cond_number:.10e}</strong></p>
        <p style="margin: 5px 0 0 0; color: #e65100;">Матрица очень плохо обусловлен</p>
        </div>
        """, unsafe_allow_html=True)
    elif cond_number > 1e5:
        st.markdown(f"""
        <div style="background-color: #fff8e1; padding: 15px; border-radius: 5px; border-left: 5px solid #ffc107;">
        <h4 style="color: #ff8f00; margin: 0;">Плохая обусловленность</h4>
        <p style="margin: 5px 0 0 0; color: #ff8f00;">Число обусловленности: <strong>{cond_number:.10e}</strong></p>
        <p style="margin: 5px 0 0 0; color: #ff8f00;">Матрица плохо обусловлена</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; border-left: 5px solid #4CAF50;">
        <h4 style="color: #2e7d32; margin: 0;">Хорошая обусловленность</h4>
        <p style="margin: 5px 0 0 0; color: #2e7d32;">Число обусловленности: <strong>{cond_number:.10e}</strong></p>
        <p style="margin: 5px 0 0 0; color: #2e7d32;">Матрица хорошо обусловлена</p>
        </div>
        """, unsafe_allow_html=True)
    
    # График зависимости обусловленности от размера матрицы
    st.subheader("Зависимость обусловленности от размера матрицы")
    
    max_n_plot = st.slider("Максимальный размер матрицы для графика:", 
                          min_value=5, max_value=38, value=10, key="cond_slider")
    
    sizes, condition_numbers = compute_condition_numbers(max_n_plot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Увеличиваем отступ сверху для заголовка
    fig.subplots_adjust(top=0.85)
    
    ax.semilogy(sizes, condition_numbers, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel('Размер матрицы n')
    ax.set_ylabel('Число обусловленности (логарифмическая шкала)')
    
    # Поднимаем заголовок выше с помощью параметра y
    ax.set_title('Зависимость числа обусловленности матрицы Гильберта от размера', 
                 pad=20, y=1.05)  # pad - отступ, y - позиция по вертикали
    
    ax.grid(True, alpha=0.3)
    
    # Устанавливаем равномерные деления на оси x
    ax.set_xticks(sizes)
    
    # Добавляем аннотации для некоторых точек с смещением в сторону
    for i, (size, cond) in enumerate(zip(sizes, condition_numbers)):
        if i % 2 == 0 or size == max_n_plot:  # Аннотируем каждую вторую точку и последнюю
            # Чередуем смещение вверх и вниз для разных точек
            offset_y = 15 if i % 4 == 0 else -25
            ax.annotate(f'{cond:.1e}', (size, cond), 
                       textcoords="offset points", 
                       xytext=(0, offset_y), 
                       ha='center', 
                       fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
    
    st.pyplot(fig)
    
    st.markdown("""
    **Замечания:**
    - Число обусловленности матрицы Гильберта растет экспоненциально с увеличением размера
    - При $n = 8$ матрица становится очень плохо обусловленной
    - Это объясняет трудности при численных вычислениях с матрицами Гильберта больших размеров
    """)


elif main_section == "Реализация и сравнение":
    
    st.markdown("### Реализация разложения Холецкого на Python")
    st.code("""
def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i+1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L
    """, language='python')
    
    st.markdown("### Реализация с использованием SciPy")
    st.code("""
# Использование встроенной функции SciPy
L_scipy = scipy.linalg.cholesky(H, lower=True)
det_scipy = np.linalg.det(H)
    """, language='python')
    
    # Вычисление для демонстрации
    n = st.slider("Выберите размер матрицы Гильберта n:", min_value=2, max_value=10, value=5, key="implementation_slider")
    
    H = hilbert_matrix(n)
    
    # Функция для пошаговой демонстрации
    def cholesky_step_by_step(A):
        """Пошаговое выполнение разложения Холецкого с объяснениями"""
        n = A.shape[0]
        L = np.zeros_like(A, dtype=float)
        
        st.write("**Процесс разложения:**")
        
        for i in range(n):
            for j in range(i + 1):
                # Вычисляем сумму скалярных произведений
                s = np.dot(L[i, :j], L[j, :j])
                
                if i == j:
                    # Диагональный элемент
                    L[i, j] = np.sqrt(A[i, i] - s)
                    st.latex(f"L_{{{i+1}{i+1}}} = \\sqrt{{A_{{{i+1}{i+1}}} - \\sum_{{k=1}}^{{{i}}} L_{{{i+1}k}} L_{{{i+1}k}}}} = \\sqrt{{{A[i, i]:.6f} - {s:.6f}}} = {L[i, j]:.6f}")
                else:
                    # Внедиагональный элемент
                    L[i, j] = (A[i, j] - s) / L[j, j]
                    st.latex(f"L_{{{i+1}{j+1}}} = \\frac{{A_{{{i+1}{j+1}}} - \\sum_{{k=1}}^{{{j}}} L_{{{i+1}k}} L_{{{j+1}k}}}}{{L_{{{j+1}{j+1}}}}} = \\frac{{{A[i, j]:.6f} - {s:.6f}}}{{{L[j, j]:.6f}}} = {L[i, j]:.6f}")
        
        return L

    # Кнопка для пошаговой демонстрации
    if st.button("Показать пошаговое разложение"):
        st.subheader("Пошаговое разложение Холецкого")
        L_step_by_step = cholesky_step_by_step(H)
        st.write("**Итоговая матрица L:**")
        st.write(L_step_by_step)
    
    L_custom = cholesky_decomposition(H)
    det_custom = np.prod(np.diag(L_custom)) ** 2

    L_scipy = scipy.linalg.cholesky(H, lower=True)
    det_scipy = np.linalg.det(H)
    
    st.subheader("Результаты вычислений")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Реализация на Python:**")
        st.write("Нижняя треугольная матрица L:")
        st.write(L_custom)
        # Форматирование определителя для лучшей читаемости
        if abs(det_custom) < 1e-10:
            st.write(f"Определитель: {det_custom:.20f}")
        else:
            st.write(f"Определитель: {det_custom:.15e}")
    
    with col2:
        st.write("**Реализация с SciPy:**")
        st.write("Нижняя треугольная матрица L:")
        st.write(L_scipy)
        # Форматирование определителя для лучшей читаемости
        if abs(det_scipy) < 1e-10:
            st.write(f"Определитель: {det_scipy:.20f}")
        else:
            st.write(f"Определитель: {det_scipy:.15e}")
    
    st.subheader("Сравнение результатов")
    
    # Вычисление разниц
    L_diff = np.max(np.abs(L_custom - L_scipy))
    det_diff = np.abs(det_custom - det_scipy)
    
    # Форматирование разниц для лучшей читаемости
    if L_diff < 1e-15:
        st.write(f"Разница между матрицами L (Реализация и SciPy): {L_diff:.20f}")
    else:
        st.write(f"Разница между матрицами L (Реализация и SciPy): {L_diff:.15e}")
    
    if det_diff < 1e-20:
        st.write(f"Разница между определителями: {det_diff:.25f}")
    else:
        st.write(f"Разница между определителями: {det_diff:.15e}")
