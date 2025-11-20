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
    sol = solve_ivp(f, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8)
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
        # "Сравнение методов"
    ]
    selected_section = st.radio("Перейти к разделу:", sections)

if selected_section == "Постановка задачи":
    st.header("Постановка задачи")

    st.markdown(r"""
Рассматривается задача Коши (Модель Лотка-Вольтерра):

$\begin{aligned}
  \frac{d y_1}{dt} = y_1 - y_1 y_2,
  \quad \frac{d y_2}{dt} = - y_2 + y_1 y_2,
  \quad 0 < t \leq 10,
\end{aligned}$

$\begin{aligned}
  y_1(0) = 2,
  \quad y_2(0) = 2 .
\end{aligned}$

Необходимо численно решить эту систему с использованием двухслойной схемы с весом при решении
системы нелинейных уравнений на новом временном слое методом Ньютона.
    """)
elif selected_section == "Теория":
    st.subheader("Теория")

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
    sol = solve_ivp(f, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8)
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
    pass
