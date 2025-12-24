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
        "Сравнение методов"
    ]
    selected_section = st.radio("Перейти к разделу:", sections)

if selected_section == "Постановка задачи":
    st.header("Постановка задачи")

    st.markdown(r"""
    Напишите программу для численного решения задачи Коши для
    системы обыкновенных дифференциальных уравнений
    явным методом Рунге-Кутта четвертого порядка.

    Продемонстрируйте работоспособность этой программы
    при решении задачи Коши

    $\begin{aligned}
     \frac{d^2 u }{dt^2} = - \sin(u),
     \quad 0 < t < 4 \pi ,
     \end{aligned}$

    $\begin{aligned}
      u(0) = 1,
     \quad \frac{d u}{dt} (0) = 0.
     \end{aligned}$

    Решите также эту задачу с помощью библиотеки SciPy.
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
    # ax.plot(t, y[:, 1], label = 'v(t)')
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
    # ax.plot(t, y[:, 1], label = 'v(t)')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.metric("Время выполнения", f"{computation_time:.6f} сек")

elif selected_section == "Сравнение методов":
    st.header("Сравнение собственной реализации и SciPy")

    # --- Собственная реализация ---
    start_time = time.perf_counter()
    t_own, y_own = own_solve(pendulum_system, rng, y0, 1e-2)
    time_own = time.perf_counter() - start_time

    # Интерполяция собственного решения к общей сетке
    y_own_interp = np.zeros((len(t_eval), 2))
    y_own_interp[:, 0] = np.interp(t_eval, t_own, y_own[:, 0])
    y_own_interp[:, 1] = np.interp(t_eval, t_own, y_own[:, 1])

    # --- SciPy ---
    start_time = time.perf_counter()
    t_sci, y_sci = solve_with_scipy(
        pendulum_system, rng, y0, t_eval
    )
    time_sci = time.perf_counter() - start_time

    # --- Оценка погрешности ---
    diff = np.abs(y_own_interp - y_sci)
    max_diff_u = np.max(diff[:, 0])
    max_diff_v = np.max(diff[:, 1])

    # --- Вывод метрик ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Время RK4 (собств.)", f"{time_own:.6f} сек")
        st.metric("Макс. разница u(t)", f"{max_diff_u:.2e}")
    with col2:
        st.metric("Время SciPy RK45", f"{time_sci:.6f} сек")
    #     st.metric("Макс. разница v(t)", f"{max_diff_v:.2e}")

    fig_u, ax_u = plt.subplots()
    ax_u.plot(t_eval, y_sci[:, 0], label="SciPy", linewidth=2)
    ax_u.plot(t_eval, y_own_interp[:, 0], "--", label="Собств.")
    ax_u.set_title("Сравнение решений")
    ax_u.set_xlabel("t")
    ax_u.set_ylabel("u")
    ax_u.grid()
    ax_u.legend()
    st.pyplot(fig_u)

    # fig_v, ax_v = plt.subplots()
    # ax_v.plot(t_eval, y_sci[:, 1], label="SciPy", linewidth=2)
    # ax_v.plot(t_eval, y_own_interp[:, 1], "--", label="Собств.")
    # ax_v.set_title("Сравнение решений v(t)")
    # ax_v.set_xlabel("t")
    # ax_v.set_ylabel("v")
    # ax_v.grid()
    # ax_v.legend()
    # st.pyplot(fig_v)
