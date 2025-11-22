import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import math

class OptimizationMethods:
    @staticmethod
    def golden_section_search(x, h, tol=1e-5):
        """
        Метод золотого сечения для одномерной минимизации
        Ищем alpha: argmin f(x + alpha * h)
        """
        phi = (math.sqrt(5) - 1) / 2  # Обратное число золотого сечения
        a, b = -10, 10  # Интервал поиска
        x1_val = a + (1 - phi) * (b - a)
        x2_val = a + phi * (b - a)

        y1 = [x[i] + x1_val * h[i] for i in range(len(x))]
        y2 = [x[i] + x2_val * h[i] for i in range(len(x))]
        f1 = OptimizationMethods.rosenbrock(y1)
        f2 = OptimizationMethods.rosenbrock(y2)

        while abs(b - a) > tol:
            if f1 < f2:
                b = x2_val
                x2_val = x1_val
                f2 = f1
                x1_val = a + (1 - phi) * (b - a)
                y1 = [x[i] + x1_val * h[i] for i in range(len(x))]
                f1 = OptimizationMethods.rosenbrock(y1)
            else:
                a = x1_val
                x1_val = x2_val
                f1 = f2
                x2_val = a + phi * (b - a)
                y2 = [x[i] + x2_val * h[i] for i in range(len(x))]
                f2 = OptimizationMethods.rosenbrock(y2)

        return (a + b) / 2

    @staticmethod
    def rosenbrock(x):
        """Функция Розенброка: 10(x₂ - x₁²)² + (1 - x₁)²"""
        return 10 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    @staticmethod
    def rosenbrock_grad(x):
        """Вычисление антиградиента функции f в точке x"""
        df_dx1 = -40 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        df_dx2 = 20 * (x[1] - x[0]**2)
        return [-df_dx1, -df_dx2]  # Антирадиент

    @staticmethod
    def norm(x, x1):
        """Вычисление нормы x - x1"""
        return math.sqrt(sum((x[i] - x1[i])**2 for i in range(len(x))))

    @staticmethod
    def norma(grad):
        """Вычисление нормы grad"""
        return math.sqrt(sum(g**2 for g in grad))

    @staticmethod
    def gradient_descent_golden(x0, tol=1e-6, max_iter=1000):
        """
        Градиентный метод с выбором шага методом золотого сечения
        """
        k = 0
        n = len(x0)
        
        # Начальная точка
        x1 = x0.copy()
        x = [x1[i] + 1 for i in range(n)]  # начальная точка x
        
        trajectory = [x1.copy()]
        f_values = [OptimizationMethods.rosenbrock(x1)]
        grad_norms = []
        alphas = []
        point_differences_x1 = []  # Разности по координате x₁
        point_differences_x2 = []  # Разности по координате x₂
        
        # Вычисляем начальный градиент
        h = OptimizationMethods.rosenbrock_grad(x1)
        grad_norms.append(OptimizationMethods.norma(h))
        
        while (k < max_iter and
               OptimizationMethods.norma(h) > tol and
               (k == 0 or (abs(f_values[-1] - f_values[-2]) > tol/1000 and 
                          OptimizationMethods.norm(trajectory[-1], trajectory[-2]) > tol/1000))):
            
            for i in range(n):
                x[i] = x1[i]
            
            h = OptimizationMethods.rosenbrock_grad(x1)
            alpha = OptimizationMethods.golden_section_search(x1, h, tol/10)
            
            for i in range(n):
                x1[i] = x[i] + alpha * h[i]
            
            # Вычисляем разности между текущей и предыдущей точкой по каждой координате
            if len(trajectory) > 0:
                diff_x1 = abs(x1[0] - trajectory[-1][0])
                diff_x2 = abs(x1[1] - trajectory[-1][1])
                point_differences_x1.append(diff_x1)
                point_differences_x2.append(diff_x2)
            
            trajectory.append(x1.copy())
            f_values.append(OptimizationMethods.rosenbrock(x1))
            grad_norms.append(OptimizationMethods.norma(h))
            alphas.append(alpha)
            
            k += 1
        
        return x1, OptimizationMethods.rosenbrock(x1), trajectory, f_values, alphas, grad_norms, point_differences_x1, point_differences_x2

def main():
    st.set_page_config(
        page_title="Минимизация функций - Задание 5.2", 
        layout="wide"
    )
    
    st.title("Задание 5.2: Минимизация функции нескольких переменных")
    
    # Постановка задачи
    with st.expander("Постановка задачи", expanded=True):
        st.write("""
        **Цель:** Написать программу для нахождения минимума функции нескольких переменных **градиентным методом** 
        при выборе итерационных параметров из минимума функции одной переменной, который находится **методом золотого сечения**.
        """)
        
        st.subheader("Математическая формулировка")
        st.latex(r"""
        \begin{aligned}
        &\text{Найти: } \min f(\mathbf{x}), \quad \mathbf{x} \in \mathbb{R}^n \\
        &\text{Алгоритм: } \mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \nabla f(\mathbf{x}_k) \\
        &\text{где: } \alpha_k = \arg\min_{\alpha>0} f(\mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k))
        \end{aligned}
        """)
        
        st.subheader("Тестовая функция")
        st.latex(r"""
        f(x_1, x_2) = 10(x_2 - x_1^2)^2 + (1 - x_1)^2
        """)
        st.write("""
        **Функция Розенброка** - классическая тестовая функция в оптимизации, известная своей овражной структурой 
        и нетривиальной поверхностью для градиентных методов.
        """)
    
    st.header("Градиентный метод с оптимальным шагом (метод золотого сечения)")
    
    # Математические формулы
    with st.expander("Математические основы методов"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Метод золотого сечения")
            st.latex(r"""
            \begin{aligned}
            &\text{Золотое сечение: } \varphi = \frac{\sqrt{5} - 1}{2} \\
            &x_1 = a + (1 - \varphi)(b - a) \\
            &x_2 = a + \varphi(b - a) \\
            &\text{Если } f(x_1) < f(x_2): b = x_2 \\
            &\text{Иначе: } a = x_1
            \end{aligned}
            """)
            
        with col2:
            st.subheader("Градиентный метод")
            st.latex(r"""
            \begin{aligned}
            &\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \nabla f(\mathbf{x}_k) \\
            &\alpha_k = \arg\min_{\alpha>0} f(\mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)) \\
            &\text{где } \alpha_k \text{ находится методом золотого сечения}
            \end{aligned}
            """)
        
        st.subheader("Функция Розенброка")
        st.latex(r"""
        \begin{aligned}
        &f(x_1, x_2) = 10(x_2 - x_1^2)^2 + (1 - x_1)^2 \\
        &\nabla f(x_1, x_2) = 
        \begin{bmatrix}
        -40x_1(x_2 - x_1^2) - 2(1 - x_1) \\
        20(x_2 - x_1^2)
        \end{bmatrix}
        \end{aligned}
        """)
    
    # Параметры метода
    st.sidebar.header("Параметры метода")
    
    x10 = st.sidebar.number_input("Начальная точка x₁", value=1.0, format="%.6f")
    x20 = st.sidebar.number_input("Начальная точка x₂", value=0.0, format="%.6f")
    tol = st.sidebar.number_input("Точность", value=1e-4, format="%.0e")
    max_iter = st.sidebar.slider("Максимальное число итераций", 10, 1000, 100)
    
    # Код реализации методов
    with st.expander("Код реализации метода золотого сечения"):
        st.code("""
def golden_section_search(x, h, tol=1e-5):
    phi = (math.sqrt(5) - 1) / 2  # Обратное число золотого сечения
    a, b = -10, 10  # Интервал поиска
    x1_val = a + (1 - phi) * (b - a)
    x2_val = a + phi * (b - a)

    y1 = [x[i] + x1_val * h[i] for i in range(len(x))]
    y2 = [x[i] + x2_val * h[i] for i in range(len(x))]
    f1 = f(y1)
    f2 = f(y2)

    while abs(b - a) > tol:
        if f1 < f2:
            b = x2_val
            x2_val = x1_val
            f2 = f1
            x1_val = a + (1 - phi) * (b - a)
            y1 = [x[i] + x1_val * h[i] for i in range(len(x))]
            f1 = f(y1)
        else:
            a = x1_val
            x1_val = x2_val
            f1 = f2
            x2_val = a + phi * (b - a)
            y2 = [x[i] + x2_val * h[i] for i in range(len(x))]
            f2 = f(y2)

    return (a + b) / 2
        """, language='python')
    
    with st.expander("Код реализации градиентного метода"):
        st.code("""
def gradient_descent_golden(x0, tol=1e-6, max_iter=1000):
    k = 0
    n = len(x0)
    
    # Начальная точка
    x1 = x0.copy()
    x = [x1[i] + 1 for i in range(n)]
    
    trajectory = [x1.copy()]
    f_values = [f(x1)]
    grad_norms = []
    alphas = []
    point_differences_x1 = []  # Разности по координате x₁
    point_differences_x2 = []  # Разности по координате x₂
    
    # Вычисляем начальный антиградиент
    h = grad_f(x1)
    grad_norms.append(norma(h))
    
    while (k < max_iter and
           norma(h) > tol and
           (k == 0 or (abs(f_values[-1] - f_values[-2]) > tol/1000 and 
                      norm(trajectory[-1], trajectory[-2]) > tol/1000))):
        
        for i in range(n):
            x[i] = x1[i]
        
        h = grad_f(x1)
        alpha = golden_section_search(x1, h, tol/10)
        
        for i in range(n):
            x1[i] = x[i] + alpha * h[i]
        
        # Вычисляем разности между текущей и предыдущей точкой по каждой координате
        if len(trajectory) > 0:
            diff_x1 = abs(x1[0] - trajectory[-1][0])
            diff_x2 = abs(x1[1] - trajectory[-1][1])
            point_differences_x1.append(diff_x1)
            point_differences_x2.append(diff_x2)
        
        trajectory.append(x1.copy())
        f_values.append(f(x1))
        grad_norms.append(norma(h))
        alphas.append(alpha)
        
        k += 1
    
    return x1, f(x1), trajectory, f_values, alphas, grad_norms, point_differences_x1, point_differences_x2
        """, language='python')
    
    if st.sidebar.button("Запустить оптимизацию"):
        with st.spinner("Выполняется оптимизация..."):
            # Проверка корректности параметров
            if tol <= 0:
                st.error("Ошибка: Точность должна быть положительным числом")
                return
            
            # Наш метод
            start_time_our = time.time()
            x_opt_our, f_opt_our, trajectory_our, f_values_our, alphas_our, grad_norms_our, point_differences_x1_our, point_differences_x2_our = (
                OptimizationMethods.gradient_descent_golden([x10, x20], tol, max_iter)
            )
            time_our = time.time() - start_time_our
            
            # SciPy метод
            start_time_scipy = time.time()
            result_scipy = minimize(OptimizationMethods.rosenbrock, [x10, x20], method='BFGS', 
                                  options={'gtol': tol, 'maxiter': max_iter})
            time_scipy = time.time() - start_time_scipy
            
            # Траектория SciPy (аппроксимация)
            trajectory_scipy = []
            f_values_scipy = []
            # Создаем промежуточные точки для визуализации SciPy
            if hasattr(result_scipy, 'allvecs'):
                trajectory_scipy = result_scipy.allvecs
                f_values_scipy = [OptimizationMethods.rosenbrock(x) for x in trajectory_scipy]
            else:
                # Аппроксимируем траекторию для SciPy
                trajectory_scipy = [np.array([x10, x20])]
                f_values_scipy = [OptimizationMethods.rosenbrock([x10, x20])]
                # Простая аппроксимация - линейная интерполяция между начальной и конечной точкой
                n_points = min(len(trajectory_our), 50)
                for i in range(1, n_points):
                    t = i / (n_points - 1)
                    x_interp = result_scipy.x * t + np.array([x10, x20]) * (1 - t)
                    trajectory_scipy.append(x_interp)
                    f_values_scipy.append(OptimizationMethods.rosenbrock(x_interp))
                trajectory_scipy.append(result_scipy.x)
                f_values_scipy.append(result_scipy.fun)
            
            # Результаты
            st.header("Результаты минимизации")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Наш метод")
                st.metric("x₁*", f"{x_opt_our[0]:.8f}")
                st.metric("x₂*", f"{x_opt_our[1]:.8f}")
                st.metric("f(x*)", f"{f_opt_our:.8f}")
                st.metric("Итерации", len(trajectory_our) - 1)
                st.metric("Время", f"{time_our:.4f} с")
            
            with col2:
                st.subheader("SciPy BFGS")
                st.metric("x₁*", f"{result_scipy.x[0]:.8f}")
                st.metric("x₂*", f"{result_scipy.x[1]:.8f}")
                st.metric("f(x*)", f"{result_scipy.fun:.8f}")
                st.metric("Итерации", result_scipy.nit)
                st.metric("Время", f"{time_scipy:.4f} с")
            
            with col3:
                st.subheader("Сравнение")
                st.metric("Разница x₁", f"{abs(x_opt_our[0] - result_scipy.x[0]):.2e}")
                st.metric("Разница x₂", f"{abs(x_opt_our[1] - result_scipy.x[1]):.2e}")
                st.metric("Разница f(x*)", f"{abs(f_opt_our - result_scipy.fun):.2e}")
                st.metric("Отн. время", f"{time_our/time_scipy:.2f}")
            
            # Визуализация
            st.header("Визуализация процесса оптимизации")
            
            # Автоматическое определение границ для визуализации на основе траекторий
            all_points = np.vstack([trajectory_our, trajectory_scipy])
            x1_min, x1_max = all_points[:, 0].min(), all_points[:, 0].max()
            x2_min, x2_max = all_points[:, 1].min(), all_points[:, 1].max()
            
            # Добавляем отступы для лучшего обзора
            x1_range = x1_max - x1_min
            x2_range = x2_max - x2_min
            x1_min_vis = x1_min - 0.1 * x1_range
            x1_max_vis = x1_max + 0.1 * x1_range
            x2_min_vis = x2_min - 0.1 * x2_range
            x2_max_vis = x2_max + 0.1 * x2_range
            
            # Создаем сетку для 3D поверхности
            x1_vis = np.linspace(x1_min_vis, x1_max_vis, 50)
            x2_vis = np.linspace(x2_min_vis, x2_max_vis, 50)
            X1, X2 = np.meshgrid(x1_vis, x2_vis)
            Z = np.zeros_like(X1)
            
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    Z[i, j] = OptimizationMethods.rosenbrock([X1[i, j], X2[i, j]])
            
            # 3D поверхность с траекториями
            fig_3d = go.Figure()
            
            fig_3d.add_trace(go.Surface(
                x=X1, y=X2, z=Z,
                colorscale='Viridis',
                opacity=0.7,
                showscale=True,
                name='f(x)'
            ))
            
            # Траектория нашего метода
            trajectory_array_our = np.array(trajectory_our)
            z_trajectory_our = [OptimizationMethods.rosenbrock(point) for point in trajectory_our]
            
            fig_3d.add_trace(go.Scatter3d(
                x=trajectory_array_our[:, 0],
                y=trajectory_array_our[:, 1], 
                z=z_trajectory_our,
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=4, color='red'),
                name='Наш метод'
            ))
            
            # Траектория SciPy
            trajectory_array_scipy = np.array(trajectory_scipy)
            z_trajectory_scipy = [OptimizationMethods.rosenbrock(point) for point in trajectory_scipy]
            
            fig_3d.add_trace(go.Scatter3d(
                x=trajectory_array_scipy[:, 0],
                y=trajectory_array_scipy[:, 1], 
                z=z_trajectory_scipy,
                mode='lines+markers',
                line=dict(color='blue', width=4),
                marker=dict(size=4, color='blue'),
                name='SciPy BFGS'
            ))
            
            fig_3d.update_layout(
                title='3D поверхность функции Розенброка с траекториями оптимизации',
                scene=dict(
                    xaxis_title='x₁',
                    yaxis_title='x₂',
                    zaxis_title='f(x)'
                ),
                height=600,
                legend=dict(
                    x=1.15,  # Увеличил сдвиг еще больше вправо
                    y=0.5,
                    xanchor='left',
                    yanchor='middle',
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.2)',
                    borderwidth=1
                ),
                margin=dict(r=150)  # Увеличил правый отступ для легенды
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Контурный график с обеими траекториями
            fig_contour = go.Figure()
            
            fig_contour.add_trace(go.Contour(
                x=x1_vis, y=x2_vis, z=Z,
                contours=dict(showlabels=True),
                colorscale='Viridis',
                name='Контуры функции'
            ))
            
            # Траектория нашего метода
            fig_contour.add_trace(go.Scatter(
                x=trajectory_array_our[:, 0],
                y=trajectory_array_our[:, 1],
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=6, color='red'),
                name='Наш метод'
            ))
            
            # Траектория SciPy
            fig_contour.add_trace(go.Scatter(
                x=trajectory_array_scipy[:, 0],
                y=trajectory_array_scipy[:, 1],
                mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=6, color='blue'),
                name='SciPy BFGS'
            ))
            
            # Начальная точка
            fig_contour.add_trace(go.Scatter(
                x=[trajectory_array_our[0, 0]],
                y=[trajectory_array_our[0, 1]],
                mode='markers',
                marker=dict(size=12, color='green', symbol='circle'),
                name='Начальная точка'
            ))
            
            # Конечные точки
            fig_contour.add_trace(go.Scatter(
                x=[trajectory_array_our[-1, 0]],
                y=[trajectory_array_our[-1, 1]],
                mode='markers',
                marker=dict(size=12, color='darkred', symbol='star'),
                name='Конец (наш метод)'
            ))
            
            fig_contour.add_trace(go.Scatter(
                x=[trajectory_array_scipy[-1, 0]],
                y=[trajectory_array_scipy[-1, 1]],
                mode='markers',
                marker=dict(size=12, color='darkblue', symbol='star'),
                name='Конец (SciPy)'
            ))
            
            fig_contour.update_layout(
                title='Контурный график с траекториями обоих методов',
                xaxis_title='x₁',
                yaxis_title='x₂',
                height=500,
                legend=dict(
                    x=1.15,  # Увеличил сдвиг еще больше вправо
                    y=0.5,
                    xanchor='left',
                    yanchor='middle',
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.2)',
                    borderwidth=1
                ),
                margin=dict(r=150)  # Увеличил правый отступ для легенды
            )
            
            st.plotly_chart(fig_contour, use_container_width=True)
            
            # Графики сходимости
            fig_convergence = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Сходимость функции f(x)',
                    'Изменение нормы градиента',
                    'Изменение разности точек по x₁',
                    'Изменение разности точек по x₂'
                )
            )
            
            # Сходимость функции - оба метода
            fig_convergence.add_trace(
                go.Scatter(
                    x=list(range(len(f_values_our))),
                    y=f_values_our,
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    name='Наш метод'
                ),
                row=1, col=1
            )
            
            fig_convergence.add_trace(
                go.Scatter(
                    x=list(range(len(f_values_scipy))),
                    y=f_values_scipy,
                    mode='lines+markers',
                    line=dict(color='blue', width=2),
                    name='SciPy BFGS'
                ),
                row=1, col=1
            )
            
            # Норма градиента (только наш метод, так как у SciPy нет этой информации)
            fig_convergence.add_trace(
                go.Scatter(
                    x=list(range(len(grad_norms_our))),
                    y=grad_norms_our,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='‖∇f(x)‖ (наш метод)'
                ),
                row=1, col=2
            )
            
            # Разность точек по x₁
            if point_differences_x1_our:
                fig_convergence.add_trace(
                    go.Scatter(
                        x=list(range(1, len(point_differences_x1_our) + 1)),
                        y=point_differences_x1_our,
                        mode='lines+markers',
                        line=dict(color='orange', width=2),
                        name='|x₁ₖ₊₁ - x₁ₖ|'
                    ),
                    row=2, col=1
                )
            
            # Разность точек по x₂
            if point_differences_x2_our:
                fig_convergence.add_trace(
                    go.Scatter(
                        x=list(range(1, len(point_differences_x2_our) + 1)),
                        y=point_differences_x2_our,
                        mode='lines+markers',
                        line=dict(color='purple', width=2),
                        name='|x₂ₖ₊₁ - x₂ₖ|'
                    ),
                    row=2, col=2
                )
            
            fig_convergence.update_xaxes(title_text="Итерация", row=1, col=1)
            fig_convergence.update_yaxes(title_text="f(x)", row=1, col=1)
            fig_convergence.update_xaxes(title_text="Итерация", row=1, col=2)
            fig_convergence.update_yaxes(title_text="‖∇f(x)‖", row=1, col=2)
            fig_convergence.update_xaxes(title_text="Итерация", row=2, col=1)
            fig_convergence.update_yaxes(title_text="|x₁ₖ₊₁ - x₁ₖ|", row=2, col=1)
            fig_convergence.update_xaxes(title_text="Итерация", row=2, col=2)
            fig_convergence.update_yaxes(title_text="|x₂ₖ₊₁ - x₂ₖ|", row=2, col=2)
            
            fig_convergence.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig_convergence, use_container_width=True)
            
            # Детальная информация
            st.header("Детальный анализ работы методов")
            
            with st.expander("Таблица итераций нашего метода"):
                iter_data = []
                for i in range(min(15, len(trajectory_our)-1)):
                    diff_x1 = point_differences_x1_our[i] if i < len(point_differences_x1_our) else 0
                    diff_x2 = point_differences_x2_our[i] if i < len(point_differences_x2_our) else 0
                    iter_data.append({
                        'Итерация': i,
                        'x₁': f"{trajectory_our[i][0]:.6f}",
                        'x₂': f"{trajectory_our[i][1]:.6f}",
                        'f(x)': f"{f_values_our[i]:.6f}",
                        '‖∇f(x)‖': f"{grad_norms_our[i]:.6f}" if i < len(grad_norms_our) else "-",
                        '|Δx₁|': f"{diff_x1:.6f}" if i < len(point_differences_x1_our) else "-",
                        '|Δx₂|': f"{diff_x2:.6f}" if i < len(point_differences_x2_our) else "-"
                    })
                
                st.table(pd.DataFrame(iter_data))
            
            with st.expander("Сравнительная таблица методов"):
                comparison_data = {
                    'Параметр': [
                        'Алгоритм', 'x₁*', 'x₂*', 'f(x*)', 'Итерации', 
                        'Время (с)', '‖∇f(x*)‖', 'Точность x₁', 'Точность x₂'
                    ],
                    'Наш метод': [
                        'Градиентный + золотое сечение',
                        f"{x_opt_our[0]:.10f}", 
                        f"{x_opt_our[1]:.10f}",
                        f"{f_opt_our:.10f}",
                        len(trajectory_our) - 1,
                        f"{time_our:.6f}",
                        f"{OptimizationMethods.norma(OptimizationMethods.rosenbrock_grad(x_opt_our)):.2e}",
                        f"{abs(x_opt_our[0] - 1.0):.2e}",
                        f"{abs(x_opt_our[1] - 1.0):.2e}"
                    ],
                    'SciPy BFGS': [
                        'BFGS',
                        f"{result_scipy.x[0]:.10f}",
                        f"{result_scipy.x[1]:.10f}", 
                        f"{result_scipy.fun:.10f}",
                        result_scipy.nit,
                        f"{time_scipy:.6f}",
                        f"{OptimizationMethods.norma(OptimizationMethods.rosenbrock_grad(result_scipy.x)):.2e}",
                        f"{abs(result_scipy.x[0] - 1.0):.2e}",
                        f"{abs(result_scipy.x[1] - 1.0):.2e}"
                    ]
                }
                
                st.table(pd.DataFrame(comparison_data))

if __name__ == "__main__":
    main()
