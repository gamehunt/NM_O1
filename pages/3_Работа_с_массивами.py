import streamlit as st
import numpy as np

# Заголовок приложения
st.title("Операции с массивами в NumPy")

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


if menu == "numpy.reshape":
    st.header("numpy.reshape")
    st.write("Изменяет форму массива без изменения его данных.")

    url = "https://numpy.org/doc/stable/reference/generated/numpy.reshape.html"
    st.markdown(f"[Подробнее]({url})")


    code = st.text_area("Введите код для изменения формы массива:",
                        value = "e = np.array([1, 2, 3, 4, 5, 6])\narr_r = e.reshape((2, 3))", height = 100)

    if st.button("Изменить форму массива"):
        try:
            exec(code)
            st.write("Результат:")
            st.write(arr_r)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "numpy.flatten":
    st.header("numpy.flatten")
    st.write("Преобразует многомерный массив в одномерный.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.ndarray.flatten.html"
    st.markdown(f"[Подробнее]({url})")


    code = st.text_area("Введите код для преобразования массива:",
                        value="""arr = np.array([[1, 2], [3, 4]])\n
f = arr.flatten()""", height=100)


    if st.button("Преобразовать массив"):
        try:
            exec(code)
            st.write("Результат:")
            st.write(f)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "numpy.sum":
    st.header("numpy.sum")
    st.write("Суммирует элементы массива.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.sum.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для суммирования элементов массива:",
                        value="e = np.array([1, 2, 3, 4, 5, 6])\ntotal = np.sum(e)", height=100)

    if st.button("Суммировать элементы"):
        try:
            exec(code)
            st.write("Результат:")
            st.write(total)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "numpy.mean":
    st.header("numpy.mean")
    st.write("Вычисляет среднее значение элементов массива.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.mean.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для вычисления среднего значения:",
                        value="e = np.array([1, 2, 3, 4, 5, 6])\naverage = np.mean(e)", height=100)

    if st.button("Вычислить среднее"):
        try:
            exec(code)
            st.write("Результат:")
            st.write(average)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "numpy.dot":
    st.header("numpy.dot")
    st.write("Выполняет матричное умножение.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.dot.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для матричного умножения:",
                        value="""A = np.array([[1, 2], [3, 4]])\nB = np.array([[5, 6], [7, 8]])\nC = np.dot(A, B)""", height=100)

    if st.button("Выполнить матричное умножение"):
        try:
            exec(code)
            st.write("Результат:")
            st.write(C)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "numpy.sin":
    st.header("numpy.sin")
    st.write("Вычисляет синус каждого элемента массива.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.sin.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для вычисления синуса:",
                        value="""angles = np.array([0, np.pi / 2, np.pi])\nsine_values = np.sin(angles)""", height=100)


    if st.button("Вычислить синус"):
        try:
            exec(code)
            st.write("Результат:")
            st.write(sine_values)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "numpy.max":
    st.header("numpy.max")
    st.write("Находит максимальное значение в массиве.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.max.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для нахождения максимального значения:",
                        value="""e = np.array([1, 2, 3, 4, 5, 6])\n
max_value = np.max(e)""", height=100)

    if st.button("Найти максимальное значение"):
        try:
            exec(code)
            st.write("Результат:")
            st.write(max_value)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "numpy.min":
    st.header("numpy.min")
    st.write("Находит минимальное значение в массиве.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.min.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для нахождения минимального значения:",
                        value="""e = np.array([1, 2, 3, 4, 5, 6])\n
min_value = np.min(e)""", height=100)

    if st.button("Найти минимальное значение"):
        try:
            exec(code)
            st.write("Результат:")
            st.write(min_value)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "numpy.std":
    st.header("numpy.std")
    st.write("Вычисляет стандартное отклонение элементов массива.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.std.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для вычисления стандартного отклонения:",
                        value="e = np.array([1, 2, 3, 4, 5, 6])\nstd_dev = np.std(e)", height=100)

    if st.button("Вычис лить стандартное отклонение"):
        try:
            exec(code)
            st.write("Результат:")
            st.write(std_dev)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "numpy.unique":
    st.header("numpy.unique")
    st.write("Возвращает уникальные значения из массива.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.unique.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для получения уникальных значений:",
                        value="unique_values = np.unique(np.array([1, 2, 2, 3, 4, 4, 5]))", height=100)

    if st.button("Получить уникальные значения"):
        try:
            exec(code)
            st.write("Результат:")
            st.write(unique_values)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "numpy.concatenate":
    st.header("numpy.concatenate")
    st.write("Объединяет два или более массивов вдоль заданной оси.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.concatenate.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для объединения массивов:",
                        value="array1 = np.array([1, 2, 3])\narray2 = np.array([4, 5, 6])\ncombined = np.concatenate((array1, array2))", height=100)

    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6])

    if st.button("Объединить массивы"):
        try:
            exec(code)
            st.write("Результат:")
            st.write(combined)
        except Exception as e:
            st.error(f"Ошибка: {e}")

if menu == "numpy.where":
    st.header("numpy.where")
    st.write("Возвращает индексы, где условие истинно.")

    url = "https://numpy.org/doc/2.1/reference/generated/numpy.where.html"
    st.markdown(f"[Подробнее]({url})")

    code = st.text_area("Введите код для получения индексов:",
                        value="e = np.array([1, 2, 3, 4, 5, 6])\nindices = np.where(e > 3)", height=100)

    if st.button("Получить индексы"):
        try:
            exec(code)
            st.write("Результат:")
            st.write(indices)
        except Exception as e:
            st.error(f"Ошибка: {e}")