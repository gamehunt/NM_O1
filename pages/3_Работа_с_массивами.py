import streamlit as st
import numpy as np

from shared import *

# Заголовок приложения
st.title("Операции с массивами в NumPy")

# Меню
menu = st.sidebar.radio("***",
                        (
                            "numpy.reshape",
                            "numpy.flatten",
                            "numpy.sum",
                            "numpy.mean",
                            "numpy.dot",
                            "numpy.sin",
                            "numpy.max",
                            "numpy.min",
                            "numpy.std",
                            "numpy.unique",
                            "numpy.concatenate",
                            "numpy.where",
                        )
                        )

# numpy.reshape
if menu == "numpy.reshape":
    def example():
        code = st.text_area("Введите код для изменения формы массива:",
                            value="e = np.array([1, 2, 3, 4, 5, 6])\narr_r = e.reshape((2, 3))", height=100)
        
        if st.button("Изменить форму массива"):
            result = exec_and_return(code, "arr_r")
            st.write("Результат:")
            st.write(result)
    
    define_function("numpy.reshape", 
                   "Изменяет форму массива без изменения его данных.",
                   "numpy.reshape",
                   example,
                   code="numpy.reshape(a, /, shape=None, order='C', *, newshape=None, copy=None)")

# numpy.flatten
if menu == "numpy.flatten":
    def example():
        code = st.text_area("Введите код для преобразования массива:",
                            value="""arr = np.array([[1, 2], [3, 4]])\nf = arr.flatten()""", height=100)
        
        if st.button("Преобразовать массив"):
            result = exec_and_return(code, "f")
            st.write("Результат:")
            st.write(result)
    
    define_function("numpy.flatten",
                   "Преобразует многомерный массив в одномерный.",
                   "numpy.ndarray.flatten",
                    example,
                    code="ndarray.flatten(order='C')")

# numpy.sum
if menu == "numpy.sum":
    def example():
        code = st.text_area("Введите код для суммирования элементов массива:",
                            value="e = np.array([1, 2, 3, 4, 5, 6])\ntotal = np.sum(e)", height=100)
        
        if st.button("Суммировать элементы"):
            result = exec_and_return(code, "total")
            st.write("Результат:")
            st.write(result)
    
    define_function("numpy.sum",
                   "Суммирует элементы массива.",
                   "numpy.sum",
                   example,
                    code="numpy.sum(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)")

# numpy.mean
if menu == "numpy.mean":
    def example():
        code = st.text_area("Введите код для вычисления среднего значения:",
                            value="e = np.array([1, 2, 3, 4, 5, 6])\naverage = np.mean(e)", height=100)
        
        if st.button("Вычислить среднее"):
            result = exec_and_return(code, "average")
            st.write("Результат:")
            st.write(result)
    
    define_function("numpy.mean",
                   "Вычисляет среднее значение элементов массива.",
                   "numpy.mean",
                   example,
                    code="numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)")

# numpy.dot
if menu == "numpy.dot":
    def example():
        code = st.text_area("Введите код для матричного умножения:",
                            value="""A = np.array([[1, 2], [3, 4]])\nB = np.array([[5, 6], [7, 8]])\nC = np.dot(A, B)""", height=100)
        
        if st.button("Выполнить матричное умножение"):
            result = exec_and_return(code, "C")
            st.write("Результат:")
            st.write(result)
    
    define_function("numpy.dot",
                   "Выполняет матричное умножение.",
                   "numpy.dot",
                   example,
                    code="numpy.dot(a, b, out=None)")

# numpy.sin
if menu == "numpy.sin":
    def example():
        code = st.text_area("Введите код для вычисления синуса:",
                            value="""angles = np.array([0, np.pi / 2, np.pi])\nsine_values = np.sin(angles)""", height=100)
        
        if st.button("Вычислить синус"):
            result = exec_and_return(code, "sine_values")
            st.write("Результат:")
            st.write(result)
    
    define_function("numpy.sin",
                   "Вычисляет синус каждого элемента массива.",
                   "numpy.sin",
                   example,
                    code="numpy.sin(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature]) = <ufunc 'sin'>")

# numpy.max
if menu == "numpy.max":
    def example():
        code = st.text_area("Введите код для нахождения максимального значения:",
                            value="""e = np.array([1, 2, 3, 4, 5, 6])\nmax_value = np.max(e)""", height=100)
        
        if st.button("Найти максимальное значение"):
            result = exec_and_return(code, "max_value")
            st.write("Результат:")
            st.write(result)
    
    define_function("numpy.max",
                   "Находит максимальное значение в массиве.",
                   "numpy.max",
                   example,
                    code="numpy.max(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)")

# numpy.min
if menu == "numpy.min":
    def example():
        code = st.text_area("Введите код для нахождения минимального значения:",
                            value="""e = np.array([1, 2, 3, 4, 5, 6])\nmin_value = np.min(e)""", height=100)
        
        if st.button("Найти минимальное значение"):
            result = exec_and_return(code, "min_value")
            st.write("Результат:")
            st.write(result)
    
    define_function("numpy.min",
                   "Находит минимальное значение в массиве.",
                   "numpy.min",
                   example,
                    code="numpy.min(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)")

# numpy.std
if menu == "numpy.std":
    def example():
        code = st.text_area("Введите код для вычисления стандартного отклонения:",
                            value="e = np.array([1, 2, 3, 4, 5, 6])\nstd_dev = np.std(e)", height=100)
        
        if st.button("Вычислить стандартное отклонение"):
            result = exec_and_return(code, "std_dev")
            st.write("Результат:")
            st.write(result)
    
    define_function("numpy.std",
                   "Вычисляет стандартное отклонение элементов массива.",
                   "numpy.std",
                   example,
                    code="numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>, mean=<no value>, correction=<no value>)")

# numpy.unique
if menu == "numpy.unique":
    def example():
        code = st.text_area("Введите код для получения уникальных значений:",
                            value="unique_values = np.unique(np.array([1, 2, 2, 3, 4, 4, 5]))", height=100)
        
        if st.button("Получить уникальные значения"):
            result = exec_and_return(code, "unique_values")
            st.write("Результат:")
            st.write(result)
    
    define_function("numpy.unique",
                   "Возвращает уникальные значения из массива.",
                   "numpy.unique",
                   example,
                    code="numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True, sorted=True)")

# numpy.concatenate
if menu == "numpy.concatenate":
    def example():
        code = st.text_area("Введите код для объединения массивов:",
                            value="array1 = np.array([1, 2, 3])\narray2 = np.array([4, 5, 6])\ncombined = np.concatenate((array1, array2))", height=100)
        
        if st.button("Объединить массивы"):
            result = exec_and_return(code, "combined")
            st.write("Результат:")
            st.write(result)
    
    define_function("numpy.concatenate",
                   "Объединяет два или более массивов вдоль заданной оси.",
                   "numpy.concatenate",
                   example,
                    code="numpy.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting='same_kind')")

# numpy.where
if menu == "numpy.where":
    def example():
        code = st.text_area("Введите код для получения индексов:",
                            value="e = np.array([1, 2, 3, 4, 5, 6])\nindices = np.where(e > 3)", height=100)
        
        if st.button("Получить индексы"):
            result = exec_and_return(code, "indices")
            st.write("Результат:")
            st.write(result)
    
    define_function("numpy.where",
                   "Возвращает индексы, где условие истинно.",
                   "numpy.where",
                   example,
                    code="numpy.where(condition, [x, y, ]/)")
