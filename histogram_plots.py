import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def histograms_x2(tensor1, tensor2):
    """
    Crea los histogramas de las listas pasadas,
    con anotaciones y líneas marcadas.

    :param tensor1: Primera lista
    :param tensor2: Segunda lista
    :return: None
    """

    # Promedio de los valores de las listas pasadas
    promedio_valores_1 = promedio_valores(tensor1)
    promedio_valores_2 = promedio_valores(tensor2)

    # Crea la figura que va a contener ambos histogramas
    plt.figure(figsize=(15, 5))

    # Figura número 1
    plt.subplot(121)
    cantidad_valores_por_clase_1, clases_1, patches1 = plt.hist(tensor1)
    cantidad_máxima_1 = np.max(cantidad_valores_por_clase_1)
    promedio_cantidades = promedio_valores(cantidad_valores_por_clase_1)
    lista_promedio_cantidades = []
    for _ in clases_1:
        lista_promedio_cantidades.append(promedio_cantidades)
    plt.plot(list(clases_1), lista_promedio_cantidades, '--', linewidth=3)
    plt.plot((promedio_valores_1, promedio_valores_1), (0, cantidad_máxima_1 * 1.1), 'r-', linewidth=3)
    plt.xlabel('Magnitud de los valores')
    plt.ylabel('Cantidad de valores por clase')
    plt.title('Histograma {} valores'.format(len(tensor1)))
    plt.text(promedio_valores_1 * 1.05, cantidad_máxima_1,
             "Promedio valores \naleatorios creados:\n{}".format(round(promedio_valores_1, 5)),
             bbox={'facecolor': 'red', 'alpha': 0.75, 'pad': 2})
    plt.text(clases_1[0], promedio_cantidades * 0.75,
             "Promedio de cantidad\nde valores por clase:\n{}".format(promedio_cantidades),
             bbox={'facecolor': 'orange', 'alpha': 0.75, 'pad': 2})
    plt.grid()

    # Figura número 2
    plt.subplot(122)
    cantidad_valores_por_clase_2, clases_2, patches2 = plt.hist(tensor2)
    cantidad_máxima_2 = np.max(cantidad_valores_por_clase_2)
    promedio_cantidades = promedio_valores(cantidad_valores_por_clase_2)
    lista_promedio_cantidades = []
    for _ in clases_2:
        lista_promedio_cantidades.append(promedio_cantidades)
    plt.plot(list(clases_2), lista_promedio_cantidades, '--', linewidth=3)
    plt.plot((promedio_valores_2, promedio_valores_2), (0, cantidad_máxima_2 * 1.1), 'r-', linewidth=3)
    plt.xlabel('Magnitud de los valores')
    plt.ylabel('Cantidad de valores por clase')
    plt.title('Histograma {} valores'.format(len(tensor2)))
    plt.text(promedio_valores_2 * 1.05, cantidad_máxima_2,
             "Promedio valores \naleatorios creados:\n{}".format(round(promedio_valores_2, 5)),
             bbox={'facecolor': 'red', 'alpha': 0.75, 'pad': 2})
    plt.text(clases_2[0], promedio_cantidades * 0.81,
             "Promedio de cantidad\nde valores por clase:\n{}".format(promedio_cantidades),
             bbox={'facecolor': 'orange', 'alpha': 0.75, 'pad': 2})
    plt.grid()

    # plt.show()


def promedio_valores(tensor):
    """
    Calcula el promedio de los valores de la lista pasada

    :param tensor: La lista con los valores
    :return: El promedio de los valores
    """
    total = 0
    for x in tensor:
        total += x
    promedio = total / len(tensor)
    return promedio


# Creación de ambos tensores para test
tensor_100 = tf.random_uniform([100])
tensor_1000 = tf.random_uniform([1000], minval=20, maxval=80)

# Generación de los valores aleatorios
with tf.Session() as sess:
    tensor_100 = sess.run(tensor_100)
    tensor_1000 = sess.run(tensor_1000)

# Función de prueba
# histograms_x2(tensor_100, tensor_1000)
