import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time

class OptimizationMethods:
    @staticmethod
    def golden_section_search(f, a, b, tol=1e-6, max_iter=100):
        """
        Метод золотого сечения для минимизации функции одной переменной
        """
        phi = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi
        
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)
        f1 = f(x1)
        f2 = f(x2)
        
        iterations = 0
        history = []
        
        while abs(b - a) > tol and iterations < max_iter:
            history.append({
                'iteration': iterations,
                'a': a, 'b': b, 
                'x1': x1, 'f1': f1,
                'x2': x2, 'f2': f2,
                'interval': b - a
            })
            
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + resphi * (b - a)
                f1 = f(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = b - resphi * (b - a)
                f2 = f(x2)
            
            iterations += 1
        
        x_min = (a + b) / 2
        return x_min, f(x_min), iterations, history

    @staticmethod
    def gradient_descent_golden(f, grad_f, x0, tol=1e-6, max_iter=1000, alpha_bounds=(0, 1)):
        """
        Градиентный метод с выбором шага методом золотого сечения
        """
        x = np.array(x0, dtype=float)
        trajectory = [x.copy()]
        f_values = [f(x)]
        alphas = []
        grad_norms = []
        
        for i in range(max_iter):
            grad = grad_f(x)
            grad_norm = np.linalg.norm(grad)
            grad_norms.append(grad_norm)
            
            if grad_norm < tol:
                break
                
            def phi(alpha):
                return f(x - alpha * grad)
            
            alpha_opt, _, _, _ = OptimizationMethods.golden_section_search(
                phi, alpha_bounds[0], alpha_bounds[1], tol=1e-4
            )
            
            x = x - alpha_opt * grad
            trajectory.append(x.copy())
            f_values.append(f(x))
            alphas.append(alpha_opt)
            
            if i > 0 and np.linalg.norm(trajectory[-1] - trajectory[-2]) < tol/10:
                break
        
        return x, f(x), trajectory, f_values, alphas, grad_norms

def rosenbrock(x):
    """Функция Розенброка: 10(x₂ - x₁²)² + (1 - x₁)²"""
    return 10 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    """Градиент функции Розенброка"""
    df_dx1 = -40 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    df_dx2 = 20 * (x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2])

def main():
    st.set_page_config(
        page_title="Минимизация функций - Задание 5.2", 
        layout="wide"
    )
    
    st.title("Задание 5.2: Минимизация функции нескольких переменных")
    st.header("Градиентный метод с оптимальным шагом (метод золотого сечения)")
    
    # Параметры метода
    st.sidebar.header("Параметры метода")
    
    x10 = st.sidebar.number_input("Начальная точка x₁", -2.0, 2.0, -1.5, 0.1)
    x20 = st.sidebar.number_input("Начальная точка x₂", -1.0, 3.0, 2.0, 0.1)
    tol = st.sidebar.number_input("Точность", 1e-8, 1e-2, 1e-6, format="%.0e")
    max_iter = st.sidebar.slider("Максимальное число итераций", 10, 500, 100)
    alpha_min = st.sidebar.number_input("Минимальный шаг α", 0.0, 1.0, 0.0, 0.01)
    alpha_max = st.sidebar.number_input("Максимальный шаг α", 0.0, 2.0, 1.0, 0.01)
    
    # Код реализации методов
    with st.expander("Код реализации метода золотого сечения"):
        st.code("""
def golden_section_search(f, a, b, tol=1e-6, max_iter=100):
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    
    iterations = 0
    
    while abs(b - a) > tol and iterations < max_iter:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)
        
        iterations += 1
    
    x_min = (a + b) / 2
    return x_min, f(x_min), iterations
        """, language='python')
    
    with st.expander("Код реализации градиентного метода"):
        st.code("""
def gradient_descent_golden(f, grad_f, x0, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
            
        def phi(alpha):
            return f(x - alpha * grad)
        
        alpha_opt, _, _ = golden_section_search(phi, 0, 1, tol=1e-4)
        
        x = x - alpha_opt * grad
        trajectory.append(x.copy())
    
    return x, f(x), trajectory
        """, language='python')
    
    if st.sidebar.button("Запустить оптимизацию"):
        with st.spinner("Выполняется оптимизация..."):
            # Наш метод
            start_time_our = time.time()
            x_opt_our, f_opt_our, trajectory_our, f_values_our, alphas_our, grad_norms_our = (
                OptimizationMethods.gradient_descent_golden(
                    rosenbrock, rosenbrock_grad, [x10, x20], tol, max_iter, (alpha_min, alpha_max)
                )
            )
            time_our = time.time() - start_time_our
            
            # SciPy метод
            start_time_scipy = time.time()
            result_scipy = minimize(rosenbrock, [x10, x20], method='BFGS', jac=rosenbrock_grad, 
                                  options={'gtol': tol, 'maxiter': max_iter})
            time_scipy = time.time() - start_time_scipy
            
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
            
            # Создаем сетку для 3D поверхности
            x1 = np.linspace(-2, 2, 50)
            x2 = np.linspace(-1, 3, 50)
            X1, X2 = np.meshgrid(x1, x2)
            Z = np.zeros_like(X1)
            
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    Z[i, j] = rosenbrock([X1[i, j], X2[i, j]])
            
            # 3D поверхность с траекторией
            fig_3d = go.Figure()
            
            fig_3d.add_trace(go.Surface(
                x=X1, y=X2, z=Z,
                colorscale='Viridis',
                opacity=0.8,
                showscale=True,
                name='f(x)'
            ))
            
            trajectory_array = np.array(trajectory_our)
            z_trajectory = [rosenbrock(point) for point in trajectory_our]
            
            fig_3d.add_trace(go.Scatter3d(
                x=trajectory_array[:, 0],
                y=trajectory_array[:, 1], 
                z=z_trajectory,
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=4, color='red'),
                name='Траектория оптимизации'
            ))
            
            fig_3d.update_layout(
                title='3D поверхность функции Розенброка с траекторией оптимизации',
                scene=dict(
                    xaxis_title='x₁',
                    yaxis_title='x₂',
                    zaxis_title='f(x)'
                ),
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Контурный график
            fig_contour = go.Figure()
            
            fig_contour.add_trace(go.Contour(
                x=x1, y=x2, z=Z,
                contours=dict(showlabels=True),
                colorscale='Viridis',
                name='Контуры функции'
            ))
            
            fig_contour.add_trace(go.Scatter(
                x=trajectory_array[:, 0],
                y=trajectory_array[:, 1],
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=6, color='red'),
                name='Траектория'
            ))
            
            fig_contour.add_trace(go.Scatter(
                x=[trajectory_array[0, 0]],
                y=[trajectory_array[0, 1]],
                mode='markers',
                marker=dict(size=12, color='green', symbol='circle'),
                name='Начальная точка'
            ))
            
            fig_contour.add_trace(go.Scatter(
                x=[trajectory_array[-1, 0]],
                y=[trajectory_array[-1, 1]],
                mode='markers',
                marker=dict(size=12, color='blue', symbol='star'),
                name='Конечная точка'
            ))
            
            fig_contour.update_layout(
                title='Контурный график функции Розенброка с траекторией оптимизации',
                xaxis_title='x₁',
                yaxis_title='x₂',
                height=500
            )
            
            st.plotly_chart(fig_contour, use_container_width=True)
            
            # Графики сходимости
            fig_convergence = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Сходимость функции f(x)',
                    'Изменение нормы градиента',
                    'Изменение шага α',
                    'Скорость сходимости'
                )
            )
            
            # Сходимость функции
            fig_convergence.add_trace(
                go.Scatter(
                    x=list(range(len(f_values_our))),
                    y=f_values_our,
                    mode='lines+markers',
                    line=dict(color='blue', width=2),
                    name='f(x)'
                ),
                row=1, col=1
            )
            
            # Норма градиента
            fig_convergence.add_trace(
                go.Scatter(
                    x=list(range(len(grad_norms_our))),
                    y=grad_norms_our,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='‖∇f(x)‖'
                ),
                row=1, col=2
            )
            
            # Шаг α
            fig_convergence.add_trace(
                go.Scatter(
                    x=list(range(len(alphas_our))),
                    y=alphas_our,
                    mode='lines+markers',
                    line=dict(color='green', width=2),
                    name='Шаг α'
                ),
                row=2, col=1
            )
            
            # Скорость сходимости (логарифмическая)
            fig_convergence.add_trace(
                go.Scatter(
                    x=list(range(len(f_values_our))),
                    y=np.log10(np.array(f_values_our) - f_opt_our + 1e-15),
                    mode='lines',
                    line=dict(color='purple', width=2),
                    name='log(f(x) - f*)'
                ),
                row=2, col=2
            )
            
            fig_convergence.update_xaxes(title_text="Итерация", row=1, col=1)
            fig_convergence.update_yaxes(title_text="f(x)", row=1, col=1)
            fig_convergence.update_xaxes(title_text="Итерация", row=1, col=2)
            fig_convergence.update_yaxes(title_text="‖∇f(x)‖", type="log", row=1, col=2)
            fig_convergence.update_xaxes(title_text="Итерация", row=2, col=1)
            fig_convergence.update_yaxes(title_text="α", row=2, col=1)
            fig_convergence.update_xaxes(title_text="Итерация", row=2, col=2)
            fig_convergence.update_yaxes(title_text="log(f(x) - f*)", row=2, col=2)
            
            fig_convergence.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig_convergence, use_container_width=True)
            
            # Детальная информация
            st.header("Детальный анализ работы методов")
            
            with st.expander("Таблица итераций нашего метода"):
                iter_data = []
                for i in range(min(15, len(trajectory_our)-1)):
                    iter_data.append({
                        'Итерация': i,
                        'x₁': f"{trajectory_our[i][0]:.6f}",
                        'x₂': f"{trajectory_our[i][1]:.6f}",
                        'f(x)': f"{f_values_our[i]:.6f}",
                        '‖∇f(x)‖': f"{grad_norms_our[i]:.6f}" if i < len(grad_norms_our) else "-",
                        'α': f"{alphas_our[i]:.6f}" if i < len(alphas_our) else "-"
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
                        f"{np.linalg.norm(rosenbrock_grad(x_opt_our)):.2e}",
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
                        f"{np.linalg.norm(rosenbrock_grad(result_scipy.x)):.2e}",
                        f"{abs(result_scipy.x[0] - 1.0):.2e}",
                        f"{abs(result_scipy.x[1] - 1.0):.2e}"
                    ]
                }
                
                st.table(pd.DataFrame(comparison_data))
            
            with st.expander("Анализ скорости сходимости"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Наш метод")
                    st.metric("Средняя скорость сходимости", 
                             f"{(f_values_our[0] - f_opt_our)/(len(trajectory_our)-1):.6f} ед/итер")
                    st.metric("Конечная точность", f"{grad_norms_our[-1]:.2e}")
                    st.metric("Средний шаг", f"{np.mean(alphas_our):.6f}")
                
                with col2:
                    st.subheader("Эффективность")
                    if time_our < time_scipy:
                        st.success(f"Наш метод быстрее в {time_scipy/time_our:.2f} раз")
                    else:
                        st.warning(f"SciPy метод быстрее в {time_our/time_scipy:.2f} раз")
                    
                    if abs(f_opt_our - result_scipy.fun) < tol:
                        st.success("Точность методов сравнима")
                    else:
                        st.warning(f"Разница в точности: {abs(f_opt_our - result_scipy.fun):.2e}")

if __name__ == "__main__":
    main()
