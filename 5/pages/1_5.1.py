import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import time

# Настройка страницы
st.set_page_config(
    page_title="Метод золотого сечения",
    layout="wide"
)

# Стиль для заголовков
st.markdown("""
<style>
.big-header {
    font-size: 28px;
    font-weight: bold;
    color: #000000;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}
.sub-header {
    font-size: 22px;
    font-weight: bold;
    color: #2ca02c;
    margin-top: 1.2em;
    margin-bottom: 0.3em;
}
.code-block {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #4CAF50;
    font-family: monospace;
    margin: 10px 0;
}
.result-box {
    background-color: #e8f5e8;
    padding: 15px;
    border-radius: 5px;
    border: 1px solid #4CAF50;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Боковое меню
st.sidebar.title("Навигация")
sections = [
    "Метод золотого сечения",
    "Основные обозначения",
    "Постановка задачи",
    "Определение и свойства",
    "Алгоритм метода",
    "Реализация и сравнение"
]
selected_section = st.sidebar.radio("Выберите раздел:", sections)

# --- МЕТОД ЗОЛОТОГО СЕЧЕНИЯ ---
if selected_section == "Метод золотого сечения":
    st.markdown('<div class="big-header">Метод золотого сечения</div>', unsafe_allow_html=True)
    st.write("""
    Метод золотого сечения — это эффективный алгоритм одномерной оптимизации для унимодальных функций. 
    Он использует свойства золотого сечения для минимизации количества вычислений функции при поиске экстремума.
    """)

# --- Математические обозначения ---
elif selected_section == "Основные обозначения":
    st.markdown('<div class="big-header">Основные обозначения</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Математические обозначения:
    - **$f(x)$** - целевая функция, которую нужно минимизировать
    - **$[a, b]$** - начальный интервал поиска минимума
    - **$x_1^{(i)}, x_2^{(i)}$** - внутренние точки золотого сечения на итерации $i$
    - **$\\varepsilon$** - заданная абсолютная погрешность
    - **$n$** - количество итераций, необходимое для достижения точности $\\varepsilon$
    - **$\\phi = \\frac{\\sqrt{5}-1}{2} \\approx 0.618$** - коэффициент золотого сечения
    - **$x_m$** - приближенное значение точки минимума
    - **$f_m = f(x_m)$** - приближенное значение минимального значения функции
    - **$L$** - константа Липшица функции $f(x)$
    - **$\\rho(x)$** - угловой коэффициент касательной к графику функции
    
    ### Обозначения в коде:
    - **`target_function(x)`** - целевая функция для минимизации
    - **`a, b`** - границы интервала поиска [a, b]
    - **`tolerance`** - точность ε (допустимая погрешность)
    - **`phi`** - коэффициент золотого сечения φ ≈ 0.618
    - **`x1, x2`** - внутренние точки золотого сечения на текущей итерации
    - **`f1, f2`** - значения функции в точках x1 и x2
    - **`x_min`** - найденная точка минимума
    - **`f_min`** - минимальное значение функции в точке x_min
    - **`iterations`** - фактическое количество выполненных итераций
    - **`max_iter`** - максимальное допустимое количество итераций
    - **`history`** - список для хранения истории всех итераций
    - **`golden_section_search()`** - функция реализации метода золотого сечения
    - **`minimize_scalar()`** - встроенная функция SciPy для минимизации
    - **`bracket=(a, b)`** - параметр для задания интервала в SciPy
    - **`method='golden'`** - указание метода оптимизации в SciPy
    - **`result.x`** - найденная точка минимума в результате SciPy
    - **`result.fun`** - минимальное значение функции в результате SciPy
    - **`result.nfev`** - количество вычислений функции в SciPy
    - **`result.success`** - флаг успешного выполнения в SciPy
    """)

# --- ПОСТАНОВКА ЗАДАЧИ ---
elif selected_section == "Постановка задачи":
    st.markdown('<div class="big-header">Постановка задачи</div>', unsafe_allow_html=True)
    st.write("""
    Необходимо написать программу для нахождения минимума функции одной переменной $ f(x) $ на интервале $[a, b]$ методом золотого сечения.

    **Целевая функция**:
    $$
    f(x) = \\frac{x^2 - 6x + 12}{x^2 + 6x + 12}
    $$

    **Интервал поиска**: $[0, 20]$

    **Также требуется** решить задачу с помощью библиотеки SciPy для сравнения результатов.
    """)

# --- ОПРЕДЕЛЕНИЕ И СВОЙСТВА ---
elif selected_section == "Определение и свойства":
    st.markdown('<div class="big-header">Определение и свойства</div>', unsafe_allow_html=True)
    st.write("""
    В методе золотого сечения две внутренние точки, используемые для сокращения отрезка поиска, выбираются так, чтобы одна из них использовалась с той же целью и на следующем уже сокращённом отрезке. Это приводит к тому, что число вычислений функции сокращается вдвое, и на каждой итерации требуется вычисление только одного нового значения функции.

    **Определение**: Точка производит золотое сечение отрезка, если отношение длины всего отрезка к длине большей части равно отношению длины большей части к меньшей.

    На отрезке $[a, b]$ точки золотого сечения $x_1$ и $x_2$ выбираются симметрично относительно концов:
    $$
    \\frac{b-a}{b-x_1} = \\frac{b-a}{x_2-a} = \\frac{b-x_1}{x_1-a} = \\frac{x_2-a}{b-x_2}
    $$

    При этом:
    - $x_1$ — вторая точка золотого сечения отрезка $[a, b]$,
    - $x_2$ — первая точка золотого сечения отрезка $[a, b]$.

    Зная одну из точек, другую можно найти по формулам:
    $$
    x_1 = a + b - x_2, \\quad x_2 = a + b - x_1
    $$

    Пусть функция $f(x)$ унимодальна на отрезке $[a, b]$ и требуется найти точку минимума $x$ с абсолютной погрешностью $\\varepsilon > 0$.
    """)

    # Вставка изображения
    st.image("https://studfile.net/html/2706/534/html_jBr2siGE9u.AmWb/htmlconvd-cpIO3__html_d962a57e299fd452.png", caption="Рис. 1. Один из вариантов выбора отрезка [a₁, b₁]", use_container_width=True)

# --- АЛГОРИТМ МЕТОДА ---
elif selected_section == "Алгоритм метода":
    st.markdown('<div class="big-header">Алгоритм метода</div>', unsafe_allow_html=True)
    st.write("""
    **Шаг 0**: Положим $a_0 = a$, $b_0 = b$. Найдём две точки $x_1^{(0)}$ и $x_2^{(0)}$ золотого сечения отрезка $[a_0, b_0]$:

    $$
    x_1^{(0)} = a_0 + \\frac{3 - \\sqrt{5}}{2}(b_0 - a_0) = a_0 + 0.381966011(b_0 - a_0)
    $$
    $$
    x_2^{(0)} = a_0 + \\frac{\\sqrt{5} - 1}{2}(b_0 - a_0) = a_0 + 0.618033989(b_0 - a_0)
    $$

    Вычислим значения функции: $f(x_1^{(0)})$ и $f(x_2^{(0)})$.

    **Шаг 1**: Определим новый отрезок поиска $[a_1, b_1]$ и новые точки $x_1^{(1)}, x_2^{(1)}$:

    - Если $f(x_1^{(0)}) \\leq f(x_2^{(0)})$:
      $$
      a_1 = a_0, \\quad b_1 = x_2^{(0)}, \\quad x_2^{(1)} = x_1^{(0)}, \\quad x_1^{(1)} = a_1 + b_1 - x_2^{(1)}
      $$
      Приближение: $\\bar{x}_1 = x_1^{(0)}$

    - Если $f(x_1^{(0)}) > f(x_2^{(0)})$:
      $$
      a_1 = x_1^{(0)}, \\quad b_1 = b_0, \\quad x_1^{(1)} = x_2^{(0)}, \\quad x_2^{(1)} = a_1 + b_1 - x_1^{(1)}
      $$
      Приближение: $\\bar{x}_1 = x_2^{(0)}$

    **Шаг i (i ≥ 2)**: Аналогично шагу 1, но на отрезке $[a_{i-1}, b_{i-1}]$:

    - Если $f(x_1^{(i-1)}) \\leq f(x_2^{(i-1)})$:
      $$
      a_i = a_{i-1}, \\quad b_i = x_2^{(i-1)}, \\quad x_2^{(i)} = x_1^{(i-1)}, \\quad x_1^{(i)} = a_i + b_i - x_2^{(i)}
      $$
      Приближение: $\\bar{x}_i = x_1^{(i-1)}$

    - Если $f(x_1^{(i-1)}) > f(x_2^{(i-1)})$:
      $$
      a_i = x_1^{(i-1)}, \\quad b_i = b_{i-1}, \\quad x_1^{(i)} = x_2^{(i-1)}, \\quad x_2^{(i)} = a_i + b_i - x_1^{(i)}
      $$
      Приближение: $\\bar{x}_i = x_2^{(i-1)}$

    **Условие остановки**: Поиск заканчивается, когда длина интервала становится не больше заданной точности $\\varepsilon$:
    $$
    |b_n - a_n| \\leq \\varepsilon
    $$

    **Результат**: Приближённая точка минимума $x_m = \\bar{x}_n$, минимальное значение $f_m = f(x_m)$.

    **Оценка числа шагов**:
    $$
    n \\geq \\frac{\\ln\\left(\\frac{\\varepsilon}{b - a}\\right)}{\\ln\\left(\\frac{\\sqrt{5}-1}{2}\\right)} - 1 \\approx -2.11 \\cdot \\ln\\left(\\frac{\\varepsilon}{b - a}\\right) - 1
    $$

    **Оценка погрешности значения функции**:
    $$
    |f_* - f(x_m)| \\leq L |x_* - x_m| \\leq L (b_n - a_n) \\leq L \\cdot \\frac{\\sqrt{5}-1}{2} \\varepsilon
    $$
    где $L = \\max_{x \\in [a_n, b_n]} |\\rho(x)|$, а $\\rho(x) = \\tan \\alpha$ — угловой коэффициент касательной к графику функции.
    """)

# --- РЕАЛИЗАЦИЯ И СРАВНЕНИЕ ---
elif selected_section == "Реализация и сравнение":
    st.markdown('<div class="big-header">Реализация и сравнение</div>', unsafe_allow_html=True)
    
    # Определение целевой функции
    def target_function(x):
        return (x**2 - 6*x + 12) / (x**2 + 6*x + 12)
    
    # Реализация метода золотого сечения
    def golden_section_search(f, a, b, tol=1e-3, max_iter=100):
        """
        Метод золотого сечения для поиска минимума функции f на интервале [a, b]
        
        Parameters:
        f - целевая функция
        a, b - границы интервала
        tol - требуемая точность
        max_iter - максимальное количество итераций
        
        Returns:
        x_min - точка минимума
        f_min - значение функции в точке минимума
        history - история вычислений
        """
        # Золотое сечение
        phi = (np.sqrt(5) - 1) / 2  # ≈ 0.618
        
        # Инициализация
        x1 = a + (1 - phi) * (b - a)
        x2 = a + phi * (b - a)
        
        f1 = f(x1)
        f2 = f(x2)
        
        history = [{'a': a, 'b': b, 'x1': x1, 'x2': x2, 'f1': f1, 'f2': f2}]
        
        for i in range(max_iter):
            if abs(b - a) < tol:
                break
                
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + (1 - phi) * (b - a)
                f1 = f(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + phi * (b - a)
                f2 = f(x2)
                
            history.append({'a': a, 'b': b, 'x1': x1, 'x2': x2, 'f1': f1, 'f2': f2})
        
        # Возвращаем середину конечного интервала как приближение минимума
        x_min = (a + b) / 2
        f_min = f(x_min)
        
        return x_min, f_min, history, i+1
    
    # Параметры
    st.sidebar.subheader("Параметры расчета")
    a = st.sidebar.number_input("Начало интервала (a)", value=0.0, step=0.1)
    b = st.sidebar.number_input("Конец интервала (b)", value=20.0, step=0.1)
    tolerance = st.sidebar.number_input("Точность (ε)", value=1e-3, format="%.e")
    
    # Реализация метода золотого сечения
    st.markdown('<div class="sub-header" style="color: black;">Реализация метода золотого сечения на Python</div>', unsafe_allow_html=True)
    
    st.code("""
def golden_section_search(f, a, b, tol=1e-3, max_iter=100):
    \"\"\"
    Метод золотого сечения для поиска минимума функции f на интервале [a, b]
    \"\"\"
    # Золотое сечение
    phi = (np.sqrt(5) - 1) / 2  # ≈ 0.618
    
    # Инициализация
    x1 = a + (1 - phi) * (b - a)
    x2 = a + phi * (b - a)
    
    f1 = f(x1)
    f2 = f(x2)
    
    history = [{'a': a, 'b': b, 'x1': x1, 'x2': x2, 'f1': f1, 'f2': f2}]
    
    for i in range(max_iter):
        if abs(b - a) < tol:
            break
            
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - phi) * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + phi * (b - a)
            f2 = f(x2)
            
        history.append({'a': a, 'b': b, 'x1': x1, 'x2': x2, 'f1': f1, 'f2': f2})
    
    x_min = (a + b) / 2
    f_min = f(x_min)
    
    return x_min, f_min, history, i+1
""", language='python')
    
    # Реализация с помощью SciPy
    st.markdown('<div class="sub-header" style="color: black;">Реализация с помощью SciPy</div>', unsafe_allow_html=True)
    
    st.code("""
from scipy.optimize import minimize_scalar

result = minimize_scalar(target_function, 
                        bracket=(a, b), 
                        method='golden', 
                        tol=tolerance)

x_min_scipy = result.x
f_min_scipy = result.fun
""", language='python')
    
    # Выполнение расчетов
    if st.button("Выполнить расчет"):
        # Собственная реализация
        start_time = time.time()
        x_min_custom, f_min_custom, history, iterations = golden_section_search(target_function, a, b, tolerance)
        custom_time = time.time() - start_time
        
        # Реализация SciPy - метод golden
        start_time = time.time()
        result_golden = minimize_scalar(target_function, bracket=(a, b), method='golden', tol=tolerance)
        scipy_golden_time = time.time() - start_time
        x_min_golden = result_golden.x
        f_min_golden = result_golden.fun
        
        # Результаты вычислений
        st.markdown('<div class="sub-header" style="color: black;">Результаты вычислений</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Собственная реализация")
            st.write(f"**Точка минимума:** {x_min_custom:.8f}")
            st.write(f"**Значение функции:** {f_min_custom:.8f}")
            st.write(f"**Количество итераций:** {iterations}")
            st.write(f"**Время выполнения:** {custom_time:.6f} сек")
        
        with col2:
            st.subheader("SciPy")
            st.write(f"**Точка минимума:** {x_min_golden:.8f}")
            st.write(f"**Значение функции:** {f_min_golden:.8f}")
            st.write(f"**Количество итераций:** {result_golden.nfev}")
            st.write(f"**Время выполнения:** {scipy_golden_time:.6f} сек")
        
        # Сравнение результатов
        st.markdown('<div class="sub-header" style="color: black;">Сравнение результатов</div>', unsafe_allow_html=True)
        
        diff_x = abs(x_min_custom - x_min_golden)
        diff_f = abs(f_min_custom - f_min_golden)
        
        st.write(f"**Разница в точке минимума:** {diff_x:.15f}")
        st.write(f"**Разница в значении функции:** {diff_f:.15f}")
        
        if diff_x < tolerance and diff_f < tolerance:
            st.success("Результаты совпадают в пределах заданной точности!")
        else:
            st.warning("Результаты отличаются больше чем на заданную точность")
        
        # Анализ результатов
        st.markdown('<div class="sub-header" style="color: black;">Анализ результатов</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Почему собственная реализация часто быстрее для простых задач?
        
        **1. Минимализм против универсальности**
        - **Собственная реализация**: Только базовый алгоритм, без лишних проверок
        - **SciPy**: Множество дополнительных проверок, обработка граничных случаев, сбор диагностики
        
        **2. Прямые вызовы против оберток**
        - **Собственная реализация**: Прямые вызовы функции, минимальные накладные расходы
        - **SciPy**: Обертки для безопасности, проверка типов, обработка исключений
        
        **3. Простота против надежности**
        - **Собственная реализация**: Простые условия сходимости, быстрая работа
        - **SciPy**: Сложные критерии остановки, проверка стагнации, анализ прогресса
        
        **4. Отсутствие overhead**
        - **Собственная реализация**: Нет создания сложных объектов результатов
        - **SciPy**: Создание сложных объектов типа `OptimizeResult` с метаданными и диагностикой
        
        **5. Адаптация под конкретную задачу**
        - **Собственная реализация**: Оптимизирована под конкретную функцию и интервал
        - **SciPy**: Универсальный алгоритм для любых функций и условий
        
        ### Когда использовать собственную реализацию?
         - Простая унимодальная функция 
         - Известен интервал поиска 
         - Нужна максимальная скорость
         - Образовательные цели
        
        ### Когда лучше использовать SciPy?
        - Сложные функции с особенностями  
        - Неизвестен интервал поиска  
        - Нужна надежность и диагностика  
        - Промышленное применение
        - Автоматическая обработка ошибок  
        
        ### Вывод
        Для **учебных и простых задач** собственная реализация эффективнее из-за минимализма.  
        Для **сложных и промышленных задач** SciPy надежнее благодаря комплексной обработке всех исходов.
        """)
        
        # График функции
        st.markdown('<div class="sub-header" style="color: black;">График целевой функции</div>', unsafe_allow_html=True)
        
        # Создаем три графика вместо двух
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        
        # График 1: Общий вид на отрезке [-20, 20]
        x_plot_full = np.linspace(-20, 20, 1000)
        y_plot_full = target_function(x_plot_full)
        
        ax1.plot(x_plot_full, y_plot_full, 'b-', linewidth=2, label='f(x)')
        ax1.axvline(x_min_custom, color='red', linestyle='--', alpha=0.7, label=f'Минимум: {x_min_custom:.4f}')
        ax1.axvline(a, color='black', linestyle=':', alpha=1.0, label=f'Интервал: [{a}, {b}]')
        ax1.axvline(b, color='black', linestyle=':', alpha=1.0)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('Общий вид функции на [-20, 20]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Вид на интервале поиска [a, b]
        x_plot_interval = np.linspace(a, b, 1000)
        y_plot_interval = target_function(x_plot_interval)
        
        ax2.plot(x_plot_interval, y_plot_interval, 'b-', linewidth=2, label='f(x)')
        ax2.axvline(x_min_custom, color='red', linestyle='--', alpha=0.7, label=f'Custom: {x_min_custom:.4f}')
        ax2.axvline(x_min_golden, color='green', linestyle='--', alpha=0.7, label=f'SciPy: {x_min_golden:.4f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.set_title(f'Функция на интервале поиска [{a}, {b}]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # График 3: Увеличенный вид вокруг минимума
        margin = (b - a) * 0.1
        x_zoom = np.linspace(x_min_custom - margin, x_min_custom + margin, 500)
        y_zoom = target_function(x_zoom)
        
        ax3.plot(x_zoom, y_zoom, 'b-', linewidth=2, label='f(x)')
        ax3.axvline(x_min_custom, color='red', linestyle='--', alpha=0.7, label=f'Custom: {x_min_custom:.8f}')
        ax3.axvline(x_min_golden, color='green', linestyle='--', alpha=0.7, label=f'SciPy: {x_min_golden:.8f}')
        ax3.set_xlabel('x')
        ax3.set_ylabel('f(x)')
        ax3.set_title('Увеличенный вид вокруг минимума')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # История итераций
        st.markdown('<div class="sub-header" style="color: black;">История итераций</div>', unsafe_allow_html=True)
        
        if len(history) > 0:
            # Берем первые 5 и последние 5 итераций
            first_5 = history[:min(5, len(history))]
            last_5 = history[-min(5, len(history)):]
            
            history_data = []
            
            # Добавляем первые 5 итераций
            for i, step in enumerate(first_5):
                history_data.append({
                    'Итерация': i,
                    'a': f"{step['a']:.6f}",
                    'b': f"{step['b']:.6f}",
                    'x1': f"{step['x1']:.6f}",
                    'x2': f"{step['x2']:.6f}",
                    'f(x1)': f"{step['f1']:.6f}",
                    'f(x2)': f"{step['f2']:.6f}",
                    'Длина интервала': f"{step['b'] - step['a']:.6f}"
                })
            
            # Добавляем разделитель если есть и первые и последние итерации
            if len(history) > 5:
                history_data.append({
                    'Итерация': '...',
                    'a': '...',
                    'b': '...',
                    'x1': '...',
                    'x2': '...',
                    'f(x1)': '...',
                    'f(x2)': '...',
                    'Длина интервала': '...'
                })
                
                # Добавляем последние 5 итераций
                start_idx = len(history) - 5
                for i, step in enumerate(last_5):
                    history_data.append({
                        'Итерация': start_idx + i,
                        'a': f"{step['a']:.6f}",
                        'b': f"{step['b']:.6f}",
                        'x1': f"{step['x1']:.6f}",
                        'x2': f"{step['x2']:.6f}",
                        'f(x1)': f"{step['f1']:.6f}",
                        'f(x2)': f"{step['f2']:.6f}",
                        'Длина интервала': f"{step['b'] - step['a']:.6f}"
                    })
            
            st.table(history_data)
