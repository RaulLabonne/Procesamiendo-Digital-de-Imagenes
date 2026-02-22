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
    vecinos = []

    for i in range(fila):
        if imagen[i][col] == 255:
            if imagen[i][col + 1] == 255:
                vecinos.append((i, col))
    print(vecinos)
    if not vecinos:
        ady4 = False
        # print("R1 no es 4-adyacente a R2")
    else:
        ady4 = True
        #print("R1 es 4-adyacente a R2")

    vecinos = []
    for i in range(fila):
        if imagen[i][col] == 255:
            diagonal_arriba = (i > 0) and (imagen[i-1][col + 1] == 255)
            diagonal_abajo = (i < fila - 1) and (imagen[i+1][col + 1] == 255)

            if diagonal_arriba or diagonal_abajo:
                vecinos.append((i, col))
    print(vecinos)
    ady8 = ady4 or bool(vecinos)
    return ady4, ady8

ady4, ady8 = adyacenciaRegiones(img_binaria, alto, 7)

mensaje_ady4 = "R1 es 4_adyacente a R2" if ady4 else "R1 no 4_es adyacente a R2"
mensaje_ady8 = "R1 es 8_adyacente a R2" if ady8 else "R1 no 8_es adyacente a R2"

print(mensaje_ady4)
print(mensaje_ady8)
