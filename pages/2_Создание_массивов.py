import streamlit as st
import numpy as np

st.title("Создание массивов с использованием NumPy")

menu = st.sidebar.radio("***",
                        (
                            "Одномерный массив",
                            "Двумерный массив",
                            "Массив нулей",
                            "Массив единиц",
                            "Массив случайных чисел",
                            "Массив с заданным диапазоном",
                            "Массив с равномерным распределением",
                        )
                        )

if menu == "Одномерный массив":
    # Создание одномерного массива
    st.header("1. Одномерный массив")

    """Функция :red[numpy.array] используется для создания массивов в библиотеке NumPy. 
    Она позволяет создавать массивы из списков, кортежей или других последовательностей, 
    а также задавать тип данных и другие параметры."""
    
    url = "https://numpy.org/doc/2.1/reference/generated/numpy.array.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для создания одномерного массива:",
                        value="array = np.array([1, 2, 3, 4, 5])", height=100)

    if st.button("Создать массив"):
        try:
            exec(code)
            st.write(array)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "Двумерный массив":
    # Создание двумерного массива
    st.header("2. Двумерный массив")
    st.write("Двумерный массив в NumPy представляет собой массив, состоящий из строк и столбцов, "
             "и может быть создан с помощью функции :red[numpy.array()]. ")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.array.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для создания двумерного массива:",
                        value="two_d_array = np.array([[1, 2, 3], [4, 5, 6]])", height=100)

    if st.button("Создать массив"):
        try:
            exec(code)
            st.write(two_d_array)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "Массив нулей":
    # Создание массива нулей
    st.header("3. Массив нулей")
    st.write(":red[numpy.zeros] — это функция в библиотеке NumPy, которая используется для создания массива, "
             "заполненного нулями.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.zeros.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для создания массива нулей:",
                        value="zeros_array = np.zeros((3, 4))", height=100)

    if st.button("Создать массив"):
        try:
            exec(code)
            st.write(zeros_array)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "Массив единиц":
    # Создание массива единиц
    st.header("4. Массив единиц")
    st.write(":red[numpy.ones] — это функция в библиотеке NumPy, аналогичная :red[numpy.zeros]. ")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.ones.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для создания массива единиц:",
                        value="ones_array = np.ones((2, 3))", height=100)

    if st.button("Создать массив"):
        try:
            exec(code)
            st.write(ones_array)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "Массив случайных чисел":
    # Создание массива с произвольными значениями
    st.header("5. Массив случайных чисел")

    st.write("numpy.random.rand - функция NumPy для создания массива случайных чисел.")

    url = "https://numpy.org/doc/2.1/reference/random/generated/numpy.random.rand.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для создания массива случайных чисел:",
                        value="random_array = np.random.rand(3, 3)", height=100)

    if st.button("Создать массив"):
        try:
            exec(code)
            st.write(random_array)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "Массив с заданным диапазоном":
    # Создание массива с заданным диапазоном
    st.header("6. Массив с заданным диапазоном")
    st.write("numpy.arrange - функция NumPy для создания массива с заданным распределением.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.arange.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для создания массива с заданным диапазоном:",
                        value="range_array = np.arange(0, 10, 2)", height=100)

    if st.button("Создать массив"):
        try:
            exec(code)
            st.write(range_array)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "Массив с равномерным распределением":
    # Создание массива с равномерным распределением
    st.header("7. Массив с равномерным распределением")
    st.write(":red[numpy.linspace] - функция в библиотеке NumPy, "
             "используется для создания массива равномерно распределенных значений в заданном интервале.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.linspace.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для создания массива с равномерным распределением:",
                        value="linspace_array = np.linspace(0, 1, 5)", height=100)

    if st.button("Создать массив"):
        try:
            exec(code)
            st.write(linspace_array)
        except Exception as e:
            st.error(f"Ошибка: {e}")
