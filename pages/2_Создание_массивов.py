import streamlit as st
from shared import *

st.title("Создание массивов с использованием NumPy")

menu = st.sidebar.radio("***",(
                            "Одномерный массив",
                            "Двумерный массив",
                            "Массив нулей",
                            "Массив единиц",
                            "Массив случайных чисел",
                            "Массив с заданным диапазоном",
                            "Массив с равномерным распределением",
                        ))


if menu == "Одномерный массив":
    def example():
        code = st.text_area("введите код для создания одномерного массива:",
                            value="array = np.array([1, 2, 3, 4, 5])", height=100)
        if st.button("Создать массив"):
            st.write(exec_and_return(code, 'array'))

    define_function("1. Одномерный массив",
                    """Функция :red[numpy.array] используется для создания массивов в библиотеке NumPy. 
                    Она позволяет создавать массивы из списков, кортежей или других последовательностей, 
                    а также задавать тип данных и другие параметры.""",
                    "numpy.array",
                    example,
                    code="numpy.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)")

if menu == "Двумерный массив":
    def example():
        code = st.text_area("Введите код для создания двумерного массива:",
                            value="two_d_array = np.array([[1, 2, 3], [4, 5, 6]])", height=100)

        if st.button("Создать массив"):
            st.write(exec_and_return(code, 'two_d_array'))

    define_function("2. Двумерный массив",
                    """Двумерный массив в NumPy представляет собой массив, состоящий из строк и столбцов, 
                    и может быть создан с помощью функции :red[numpy.array()].""",
                    "numpy.array",
                    example,
                    code="numpy.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)")

if menu == "Массив нулей":
    def example():
        code = st.text_area("Введите код для создания массива нулей:",
                            value="zeros_array = np.zeros((3, 4))", height=100)

        if st.button("Создать массив"):
            st.write(exec_and_return(code, 'zeros_array'))

    define_function("3. Массив нулей",
                    """:red[numpy.zeros] — это функция в библиотеке NumPy, которая используется для создания массива,  
                    заполненного нулями.""",
                    "numpy.zeros",
                    example,
                    code="numpy.zeros(shape, dtype=float, order='C', *, like=None)")

if menu == "Массив единиц":
    def example():
        code = st.text_area("Введите код для создания массива единиц:",
                        value="ones_array = np.ones((2, 3))", height=100)
        if st.button("Создать массив"):
            st.write(exec_and_return(code, 'ones_array'))

    define_function("4. Массив единиц",
                    ":red[numpy.ones] — это функция в библиотеке NumPy, аналогичная :red[numpy.zeros]. ",
                    "numpy.ones",
                    example,
                    code="numpy.ones(shape, dtype=None, order='C', *, device=None, like=None)")

if menu == "Массив случайных чисел":
    def example():
        code = st.text_area("Введите код для создания массива случайных чисел:",
                            value="random_array = np.random.rand(3, 3)", height=100)

        if st.button("Создать массив"):
            st.write(exec_and_return(code, 'random_array'))

    define_function("5. Массив случайных чисел",
                    ":red[numpy.random.rand] - функция NumPy для создания массива случайных чисел.",
                    "../random/generated/numpy.random.rand",
                    example,
                    code="random.rand(d0, d1, ..., dn)")

if menu == "Массив с заданным диапазоном":
    def example():
        code = st.text_area("Введите код для создания массива с заданным диапазоном:",
                            value="range_array = np.arange(0, 10, 2)", height=100)
        if st.button("Создать массив"):
            st.write(exec_and_return(code, 'range_array'))

    define_function("6. Массив с заданным диапазоном",
                    ":red[numpy.arange] - функция NumPy для создания массива с заданным распределением.",
                    "numpy.arange",
                    example,
                    code="numpy.arange([start, ]stop, [step, ]dtype=None, *, device=None, like=None)")

if menu == "Массив с равномерным распределением":
    def example():
        code = st.text_area("Введите код для создания массива с равномерным распределением:",
                            value="linspace_array = np.linspace(0, 1, 5)", height=100)
        if st.button("Создать массив"):
            st.write(exec_and_return(code, 'linspace_array'))

    define_function("7. Массив с равномерным распределением",
                    """:red[numpy.linspace] - функция в библиотеке NumPy, 
                    используется для создания массива равномерно распределенных значений в заданном интервале.""",
                    "numpy.linspace",
                    example,
                    code="numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *, device=None)")
