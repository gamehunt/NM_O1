import numpy as np
import scipy
from scipy.integrate import solve_ivp

# f - система
# t0 - начало интервала
# y0 - НУ
# h - шаг интегрирования
# n_steps - число шагов
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

own_t, own_y = own_solve(pendulum_system, rng, y0, 1e-2)
scipy_t, scipy_y = solve_with_scipy(pendulum_system, rng, y0, t_eval)
