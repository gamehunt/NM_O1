import numpy as np
import scipy.linalg
import streamlit as st

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

st.title("Разложение Холецкого и определитель матрицы Гильберта")

# Создаем боковую панель для навигации - все разделы видны сразу
st.sidebar.title("Навигация")
section = st.sidebar.radio(
    "Выберите раздел:",
    ["Условные обозначения", "Постановка задачи", "Алгоритм разложения Холецкого", "Матрица Гильберта", "Реализация и сравнение"]
)

if section == "Условные обозначения":
    st.header("Условные обозначения")
    
    st.markdown("""
    В данном приложении используются следующие обозначения:
    
    ### Математические обозначения:
    - **$A$** - исходная симметричная положительно определённая матрица
    - **$L$** - нижняя треугольная матрица в разложении Холецкого
    - **$L^T$** - транспонированная матрица $L$
    - **$H$** - матрица Гильберта
    - **$n$** - размерность матрицы
    - **$a_{ij}$** - элемент матрицы $A$ в строке $i$, столбце $j$
    - **$l_{ij}$** - элемент матрицы $L$ в строке $i$, столбце $j$
    - **$\\det(A)$** - определитель матрицы $A$
    
    ### Обозначения в коде:
    - **`A`** - входная матрица для разложения
    - **`L`** - результирующая нижняя треугольная матрица
    - **`H`** - матрица Гильберта
    - **`n`** - размер матрицы
    - **`i, j`** - индексы строк и столбцов
    - **`s`** - вспомогательная переменная для суммы произведений
    - **`cholesky_decomposition(A)`** - функция для вычисления разложения Холецкого
    - **`hilbert_matrix(n)`** - функция для генерации матрицы Гильберта
    - **`np.linalg.cholesky()`** - встроенная функция NumPy для разложения Холецкого
    - **`np.linalg.det()`** - встроенная функция NumPy для вычисления определителя
    """)

elif section == "Постановка задачи":
    st.header("Постановка задачи")
    st.markdown("""
    Необходимо написать программу, реализующую разложение Холецкого $A = LL^T$ для симметричной положительно определённой матрицы $A$ и вычисляющую определитель матрицы на основе этого разложения.

    Найдите разложение Холецкого и определитель матрицы Гильберта, для которой

    $a_{ij} = \\frac{1}{i + j - 1}, \\quad i = 1, 2, \\dots, n, \\quad j = 1, 2, \\dots, n$

    при различных $n$.

    Решите также эту задачу с помощью библиотеки Numpy.
    """)

elif section == "Алгоритм разложения Холецкого":
    st.header("Алгоритм разложения Холецкого")

    st.markdown(r"""
    ### Математические формулы разложения Холецкого

    Для симметричной положительно определённой матрицы $A$ разложение $A = LL^T$ 
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
    st.markdown("- Создаём нулевую матрицу $L$ того же размера, что и $A$")

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

elif section == "Матрица Гильберта":
    st.header("Матрица Гильберта")
    
    st.markdown("---")
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
    
    n = st.slider("Выберите размер матрицы Гильберта n:", min_value=2, max_value=10, value=5, key="hilbert_slider")

    H = hilbert_matrix(n)
    
    st.subheader("Матрица Гильберта")
    st.write(H)

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

elif section == "Реализация и сравнение":
    st.header("Реализация и сравнение")
    
    st.subheader("Реализация разложения Холецкого")
    
    st.markdown("### Реализация на Python")
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
    
    st.markdown("### Реализация с использованием NumPy")
    st.code("""
# Использование встроенной функции NumPy
L_numpy = np.linalg.cholesky(H)
det_numpy = np.linalg.det(H)
    """, language='python')
    
    # Вычисление для демонстрации
    n = st.slider("Выберите размер матрицы Гильберта n:", min_value=2, max_value=10, value=5, key="implementation_slider")
    
    H = hilbert_matrix(n)
    L_custom = cholesky_decomposition(H)
    det_custom = np.prod(np.diag(L_custom)) ** 2

    L_numpy = np.linalg.cholesky(H)
    det_numpy = np.linalg.det(H)
    
    st.subheader("Результаты вычислений")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Реализация на Python:**")
        st.write("Нижняя треугольная матрица L:")
        st.write(L_custom)
        st.write(f"Определитель: {det_custom:.10e}")
    
    with col2:
        st.write("**Реализация с NumPy:**")
        st.write("Нижняя треугольная матрица L:")
        st.write(L_numpy)
        st.write(f"Определитель: {det_numpy:.10e}")
    
    st.subheader("Сравнение результатов")
    st.write(f"Разница между матрицами L (Реализация и NumPy): {np.max(np.abs(L_custom - L_numpy)):.10e}")
    st.write(f"Разница между определителями: {np.abs(det_custom - det_numpy):.10e}")
