# Alumno: Raul Emiliano Labonne Arizmendi
# Tarea 1
# Procesamiento Digital de Imagenes
# Ejercicio 1



import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# Creamos la imagen de dimension 8 x 16

alto, ancho  = 8, 16
imagen = np.zeros((alto, ancho), dtype=np.uint8)

# Generamos pixeles aleatorios

for i in range(alto):
    for j in range(ancho):
        imagen[i, j] = random.randint(0, 255)

# Dado el valor de intensidad del pixel, determinaremos su valor en binario
umbral = 128
# Si la intensidad es menor o igual a 128 es 0, en otro caso es 255
img_binaria = np.where(imagen > umbral, 255, 0).astype('uint8')

print(img_binaria)
# Guardamos la imagen
img = Image.fromarray(img_binaria, mode='L')
img.save('imagen.png')

# Buscamos sus adyacencias en la region R1 y R2
# Basta con verificar la frontera entre ambas regiones

def adyacenciaRegiones(imagen, fila, col):
    ady4, ady8 = False, False

    # Lista para saber que pixeles tienen vecinos, ya sea 4 vecindad u 8 vecindad
    vecinos = []

    # Verificamos la 4 vecindad del pixel
    for i in range(fila):
        if imagen[i][col] == 255:
            if imagen[i][col + 1] == 255:
                vecinos.append((i, col))

    # Verificamos si hay pixeles en la lista, si esta vacia, entonces no es 4 adyacente
    if not vecinos:
        ady4 = False
    else:
        ady4 = True

    # Checamos si hay pixeles con 8 vecindad verificando solamente si tienen 4 vecindad por esquinas
    vecinos = []
    for i in range(fila):
        if imagen[i][col] == 255:
            diagonal_arriba = (i > 0) and (imagen[i-1][col + 1] == 255)
            diagonal_abajo = (i < fila - 1) and (imagen[i+1][col + 1] == 255)

            if diagonal_arriba or diagonal_abajo:
                vecinos.append((i, col))

    ady8 = ady4 or bool(vecinos)
    # Regresamos los resultados
    return ady4, ady8

ady4, ady8 = adyacenciaRegiones(img_binaria, alto, 7)

mensaje_ady4 = "R1 es 4_adyacente a R2" if ady4 else "R1 no 4_es adyacente a R2"
mensaje_ady8 = "R1 es 8_adyacente a R2" if ady8 else "R1 no 8_es adyacente a R2"

print(mensaje_ady4)
print(mensaje_ady8)
