import numpy as np
import scipy
from scipy.integrate import solve_ivp
import streamlit as st
import time
import matplotlib.pyplot as plt
    
def own_newton_solve(F, J, x0, max_iter=100, tol=1e-8):
    """
    Метод Ньютона для решения системы нелинейных уравнений
    
    Parameters:
    F - функция системы
    J - якобиан
    x0 - начальное приближение
    max_iter - максимальное число итераций
    tol - допуск
    
    Returns:
    x - решение
    iterations - число итераций
    """
    x = x0.copy()
    iterations = 0
    
    for _ in range(max_iter):
        F_val = F(x)
        if np.linalg.norm(F_val) < tol:
            break
            
        J_val = J(x)
        delta_x = np.linalg.solve(J_val, -F_val)
        x += delta_x
        iterations += 1
    
    if iterations == max_iter:
        print(f"Предупреждение: достигнуто максимальное число итераций ({max_iter})")
        
    return x, iterations

def own_solve(f, f_jac, t_span, y0, t_eval=None, theta = 0.5, n_steps=1000):
    """
    Решение задачи Коши
    
    Parameters:
    f - функция правой части системы ОДУ: f(t, y)
    f_jac - якобиан функции f: jac(t, y)
    theta - вес схемы (0.5 - Кранка-Николсон, 1 - неявный Эйлер)
    t_span - интервал времени [t0, tf]
    y0 - начальные условия
    t_eval - точки, в которых нужно вычислить решение
    n_steps - число шагов по времени
    
    Returns:
    t - массив времени
    y - массив решений
    """
    t0, tf = t_span
    if t_eval is not None:
        t = t_eval
    else:
        t = np.linspace(t0, tf, n_steps)
    
    n = len(t)
    m = len(y0)
    y = np.zeros((n, m))
    y[0] = y0
    
    # Статистика итераций
    newton_iterations = []
    
    for i in range(1, n):
        dt = t[i] - t[i-1]
        y_prev = y[i-1]
        
        # Функция для метода Ньютона: F(y) = y - y_prev - dt * [(1-theta)*f(t_prev, y_prev) + theta*f(t, y)]
        def F(y_next):
            t_prev = t[i-1]
            t_curr = t[i]
            return y_next - y_prev - dt * ((1 - theta) * f(t_prev, y_prev) + 
                                         theta * f(t_curr, y_next))
        
        # Якобиан для метода Ньютона
        def J(y_next):
            t_curr = t[i]
            return np.eye(m) - dt * theta * f_jac(t_curr, y_next)
        
        # Начальное приближение (явный метод Эйлера)
        y_guess = y_prev + dt * f(t[i-1], y_prev)
        
        # Решение методом Ньютона
        y_next, iterations = own_newton_solve(F, J, y_guess)
        newton_iterations.append(iterations)
        
        y[i] = y_next
    
    return t, y

def lotka_volterra(t, y):
    """
    Система Лотка-Вольтерра (хищник-жертва)
    y[0] - популяция жертв
    y[1] - популяция хищников
    
    Параметры:
    alpha - рождаемость жертв
    beta - смертность жертв от хищников
    gamma - смертность хищников
    delta - рождаемость хищников
    """
    alpha, beta, gamma, delta = 1.0, 1.0, 1.0, 1.0
    
    dydt = np.zeros(2)
    dydt[0] = alpha * y[0] - beta * y[0] * y[1]  # Уравнение для жертв
    dydt[1] = delta * y[0] * y[1] - gamma * y[1] # Уравнение для хищников
    
    return dydt


def lotka_volterra_jac(t, y):
    """
    Якобиан системы Лотка-Вольтерра
    """
    alpha, beta, gamma, delta = 1.0, 1.0, 1.0, 1.0
    
    J = np.zeros((2, 2))
    J[0, 0] = alpha - beta * y[1]  # df0/dy0
    J[0, 1] = -beta * y[0]         # df0/dy1
    J[1, 0] = delta * y[1]         # df1/dy0
    J[1, 1] = delta * y[0] - gamma # df1/dy1
    
    return J

def solve_with_scipy(f, t_span, y0, t_eval):
    """
    Решение с помощью SciPy
    """
    sol = solve_ivp(f, t_span, y0, t_eval=t_eval, method='RK23', rtol=1e-8)
    return sol.t, sol.y.T

rng = [0, 10]
y0 = [2, 2]
t_eval = np.linspace(rng[0], rng[1], 1000)



st.title("Задача Коши для дифференциальных уравнений")

# Боковая панель с навигацией и параметрами
with st.sidebar:
    st.header("Навигация")
    sections = [
        "Постановка задачи",
        "Теория",
        "Собственная реализация",
        "Решение через SciPy",
        "Сравнение методов"
    ]
    selected_section = st.radio("Перейти к разделу:", sections)

if selected_section == "Постановка задачи":
    st.header("Постановка задачи")

    st.markdown(r"""
Напишите программу для численного решения задачи Коши для
системы обыкновенных дифференциальных уравнений
с использованием двухслойной схемы с весом при решении
системы нелинейных уравнений на новом временном слое методом Ньютона.
Используйте эту программу для решения задачи Коши
(модель Лотка-Вольтерра)

$\begin{aligned}
  \frac{d y_1}{dt} = y_1 - y_1 y_2,
  \quad \frac{d y_2}{dt} = - y_2 + y_1 y_2,
  \quad 0 < t \leq 10,
\end{aligned}$

$\begin{aligned}
  y_1(0) = 2,
  \quad y_2(0) = 2 .
\end{aligned}$

Решите также эту задачу с помощью библиотеки SciPy.
    """)
elif selected_section == "Теория":
    st.subheader("Теория")

    st.markdown("""
    Двухслойная схема с весом (σ-схема) — это численный метод для решения систем обыкновенных 
    дифференциальных уравнений (ОДУ), который обобщает несколько классических методов через 
    введение параметра веса σ.
    """)

    st.subheader("Общая форма схемы")
    st.latex(r"""
    \frac{y^{n+1} - y^n}{\tau} = \sigma f(t^{n+1}, y^{n+1}) + (1 - \sigma) f(t^n, y^n)
    """)
    st.markdown("где:")
    st.markdown("- $y^n$ — решение на n-м временном слое")
    st.markdown("- $\\tau$ — шаг по времени")
    st.markdown("- $\\sigma$ — параметр веса ($0 \\leq \\sigma \\leq 1$)")
    st.markdown("- $f(t, y)$ — правая часть системы ДУ")

    st.subheader("Порядок точности")
    st.markdown("""
    - **Первый порядок** при σ ≠ 0.5
    - **Второй порядок** при σ = 0.5
    """)

    st.subheader("Частные случаи")

    st.markdown("σ = 0 - Явный метод Эйлера")
    st.markdown("σ = 0.5 - Метод Кранка-Николсона")
    st.markdown("σ = 1 - Неявный метод Эйлера")
    
    st.subheader("Устойчивость")
    st.markdown("""
    Схема является **безусловно устойчивой** при:
    """)
    st.latex(r"\sigma \geq \frac{1}{2}")
    st.markdown("""
    При σ < 0.5 схема условно устойчива с ограничением на шаг по времени.
    """)

elif selected_section == "Собственная реализация":
    st.code('''
def own_newton_solve(F, J, x0, max_iter=100, tol=1e-8):
    """
    Метод Ньютона для решения системы нелинейных уравнений
    
    Parameters:
    F - функция системы
    J - якобиан
    x0 - начальное приближение
    max_iter - максимальное число итераций
    tol - допуск
    
    Returns:
    x - решение
    iterations - число итераций
    """
    x = x0.copy()
    iterations = 0
    
    for _ in range(max_iter):
        F_val = F(x)
        if np.linalg.norm(F_val) < tol:
            break
            
        J_val = J(x)
        delta_x = np.linalg.solve(J_val, -F_val)
        x += delta_x
        iterations += 1
    
    if iterations == max_iter:
        print(f"Предупреждение: достигнуто максимальное число итераций ({max_iter})")
        
    return x, iterations

def own_solve(f, f_jac, t_span, y0, t_eval=None, theta = 0.5, n_steps=1000):
    """
    Решение задачи Коши
    
    Parameters:
    f - функция правой части системы ОДУ: f(t, y)
    f_jac - якобиан функции f: jac(t, y)
    theta - вес схемы (0.5 - Кранка-Николсон, 1 - неявный Эйлер)
    t_span - интервал времени [t0, tf]
    y0 - начальные условия
    t_eval - точки, в которых нужно вычислить решение
    n_steps - число шагов по времени
    
    Returns:
    t - массив времени
    y - массив решений
    """
    t0, tf = t_span
    if t_eval is not None:
        t = t_eval
    else:
        t = np.linspace(t0, tf, n_steps)
    
    n = len(t)
    m = len(y0)
    y = np.zeros((n, m))
    y[0] = y0
    
    # Статистика итераций
    newton_iterations = []
    
    for i in range(1, n):
        dt = t[i] - t[i-1]
        y_prev = y[i-1]
        
        # Функция для метода Ньютона: F(y) = y - y_prev - dt * [(1-theta)*f(t_prev, y_prev) + theta*f(t, y)]
        def F(y_next):
            t_prev = t[i-1]
            t_curr = t[i]
            return y_next - y_prev - dt * ((1 - theta) * f(t_prev, y_prev) + 
                                         theta * f(t_curr, y_next))
        
        # Якобиан для метода Ньютона
        def J(y_next):
            t_curr = t[i]
            return np.eye(m) - dt * theta * f_jac(t_curr, y_next)
        
        # Начальное приближение (явный метод Эйлера)
        y_guess = y_prev + dt * f(t[i-1], y_prev)
        
        # Решение методом Ньютона
        y_next, iterations = own_newton_solve(F, J, y_guess)
        newton_iterations.append(iterations)
        
        y[i] = y_next
    
    return t, y
    ''', language='python')

    theta = st.slider('Theta', min_value=0.0, max_value=1.0, value=0.5)

    start_time = time.perf_counter()
    t, y = own_solve(lotka_volterra, lotka_volterra_jac, rng, y0, t_eval)
    computation_time = time.perf_counter() - start_time

    fig, ax = plt.subplots()
    plt.title('График решения')
    ax.plot(t, y[:, 0], label = 'y1(t)')
    ax.plot(t, y[:, 1], label = 'y2(t)')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.metric("Время выполнения", f"{computation_time:.6f} сек")

elif selected_section == "Решение через SciPy":
    st.code('''
# f - система
# t_span - интервал
# y0 - НУ
# t_eval - моменты времени, для которых будет сохранено решение
def solve_with_scipy(f, t_span, y0, t_eval):
    """
    Решение с помощью SciPy
    """
    sol = solve_ivp(f, t_span, y0, t_eval=t_eval, method='RK23', rtol=1e-8)
    return sol.t, sol.y.T
''', language = 'python')

    start_time = time.perf_counter()
    t, y = solve_with_scipy(lotka_volterra, rng, y0, t_eval)
    computation_time = time.perf_counter() - start_time

    fig, ax = plt.subplots()
    plt.title('График решения')
    ax.plot(t, y[:, 0], label = 'y1(t)')
    ax.plot(t, y[:, 1], label = 'y2(t)')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.metric("Время выполнения", f"{computation_time:.6f} сек")

elif selected_section == "Сравнение методов":
    st.header("Сравнение методов")

    theta = st.slider("Theta", 0.0, 1.0, 0.5)

    # --- Собственный метод ---
    start_time = time.perf_counter()
    t_own, y_own = own_solve(
        lotka_volterra,
        lotka_volterra_jac,
        rng,
        y0,
        t_eval=t_eval,
        theta=theta
    )
    time_own = time.perf_counter() - start_time

    # --- SciPy ---
    start_time = time.perf_counter()
    t_sci, y_sci = solve_with_scipy(
        lotka_volterra,
        rng,
        y0,
        t_eval
    )
    time_sci = time.perf_counter() - start_time

    # --- Максимальная разница ---
    diff = np.abs(y_own - y_sci)
    max_diff_y1 = np.max(diff[:, 0])
    max_diff_y2 = np.max(diff[:, 1])

    # --- Метрики ---
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Время собственной реализации", f"{time_own:.6f} сек")
        st.metric("Макс. разница y1", f"{max_diff_y1:.2e}")

    with col2:
        st.metric("Время SciPy RK23", f"{time_sci:.6f} сек")
        st.metric("Макс. разница y2", f"{max_diff_y2:.2e}")

    # --- График y1(t) ---
    fig1, ax1 = plt.subplots()
    ax1.plot(t_sci, y_sci[:, 0], label="y₁(t) — SciPy", linewidth=2)
    ax1.plot(t_own, y_own[:, 0], "--", label="y₁(t) — θ-схема")
    ax1.set_title("Сравнение решений y₁(t)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("y₁")
    ax1.grid()
    ax1.legend()
    st.pyplot(fig1)

    # --- График y2(t) ---
    fig2, ax2 = plt.subplots()
    ax2.plot(t_sci, y_sci[:, 1], label="y₂(t) — SciPy", linewidth=2)
    ax2.plot(t_own, y_own[:, 1], "--", label="y₂(t) — θ-схема")
    ax2.set_title("Сравнение решений y₂(t)")
    ax2.set_xlabel("t")
    ax2.set_ylabel("y₂")
    ax2.grid()
    ax2.legend()
    st.pyplot(fig2)
