import numpy as np
import scipy
from scipy.integrate import solve_ivp
import streamlit as st
import matplotlib.pyplot as plt
import time

# f - система
# rng - интервал
# h - шаг интегрирования
def own_solve(f, rng, y0, h):
    y0 = np.array(y0, dtype=float)
    n_steps = int((rng[1] - rng[0]) / h)
    n_eq = len(y0)  # количество уравнений в системе
    
    # Инициализация массивов для результатов
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, n_eq))
    
    # Начальные условия
    t[0] = rng[0]
    y[0] = y0
    
    # Итерационный процесс метода Рунге-Кутта 4-го порядка
    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        
        y[i+1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t[i+1] = t[i] + h
    
    return t, y

# f - система
# t_span - интервал
# y0 - НУ
# t_eval - сетка решения
def solve_with_scipy(f, t_span, y0, t_eval):
    """
    Решение с помощью SciPy
    """
    sol = solve_ivp(f, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8)
    return sol.t, sol.y.T

def pendulum_system(t, y):
    """
    Система ОДУ для уравнения маятника: d²u/dt² = -sin(u)
    
    Преобразование в систему первого порядка:
    y[0] = u (угол)
    y[1] = du/dt (угловая скорость)
    
    du/dt = y[1]
    d²u/dt² = -sin(y[0])
    """
    u, v = y
    dudt = v
    dvdt = -np.sin(u)
    return np.array([dudt, dvdt])

rng = [0, np.pi * 4]
y0 = [1.0, 0.0]

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
    Рассматривается задача Коши:

    $\begin{aligned}
     \frac{d^2 u }{dt^2} = - \sin(u),
     \quad 0 < t < 4 \pi ,
     \end{aligned}$

    $\begin{aligned}
      u(0) = 1,
     \quad \frac{d u}{dt} (0) = 0.
     \end{aligned}$

    Которая сводится к системе:

    $\begin{aligned}
        \frac{du}{dt} = v , 
        \quad \frac{dv}{dt} = -sin(u)
    \end{aligned}$

    $\begin{aligned}
      u(0) = 1,
     \quad v(0) = 0.
     \end{aligned}$

    Необходимо численно решить эту систему явным методом Рунге-Кутты 4 порядка
    """)
elif selected_section == "Теория":
    st.subheader("Теория")

    st.markdown("""
        Метод Рунге-Кутты 4-го порядка основан на идее **аппроксимации решения 
        с помощью взвешенного среднего нескольких наклонов** (производных) на каждом шаге.
        
        Вместо использования одной производной (как в методе Эйлера), метод использует 
        4 различные оценки производной в точках внутри шага интегрирования.
        """)

    st.markdown(r"""
        ### Формула метода

        Рассмотрим задачу Коши для системы ОДУ:
        
        $$
        \begin{cases}
        \frac{dy}{dt} = f(t, y) \\
        y(t_0) = y_0
        \end{cases}
        $$

        Для шага от $t_n$ к $t_{n+1} = t_n + h$:
        
        $$
        \begin{align*}
        & k_1 = f(t_n, y_n) \\
        & k_2 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_1) \\
        & k_3 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_2) \\
        & k_4 = f(t_n + h, y_n + h k_3) \\
        & y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
        \end{align*}
        $$
        """)
        
    st.markdown(r"""
        ### Обоснование весовых коэффициентов
        
        Весовые коэффициенты $\frac{1}{6}$, $\frac{2}{6}$, $\frac{2}{6}$, $\frac{1}{6}$ 
        выбраны таким образом, чтобы метод имел 4-й порядок точности. Это означает, 
        что локальная ошибка на одном шаге пропорциональна $h^5$.
        """)
elif selected_section == "Собственная реализация":
    st.code('''
# f - система
# rng - интервал
# h - шаг интегрирования
def own_solve(f, rng, y0, h):
    n_steps = int((rng[1] - rng[0]) / h)
    n_eq = len(y0)  # количество уравнений в системе
    
    # Инициализация массивов для результатов
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, n_eq))
    
    # Начальные условия
    t[0] = rng[0]
    y[0] = y0
    
    # Итерационный процесс метода Рунге-Кутта 4-го порядка
    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        
        y[i+1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t[i+1] = t[i] + h
    
    return t, y
    ''', language='python')

    start_time = time.perf_counter()
    t, y = own_solve(pendulum_system, rng, y0, 1e-2)
    computation_time = time.perf_counter() - start_time

    fig, ax = plt.subplots()
    plt.title('График решения')
    ax.plot(t, y[:, 0], label = 'u(t)')
    ax.plot(t, y[:, 1], label = 'v(t)')
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
    t, y = solve_with_scipy(pendulum_system, rng, y0, t_eval)
    computation_time = time.perf_counter() - start_time

    fig, ax = plt.subplots()
    plt.title('График решения')
    ax.plot(t, y[:, 0], label = 'u(t)')
    ax.plot(t, y[:, 1], label = 'v(t)')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.metric("Время выполнения", f"{computation_time:.6f} сек")
