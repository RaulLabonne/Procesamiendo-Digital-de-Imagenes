
# Alumno: Raul Emiliano Labonne Arizmendi
# Tarea 1
# Procesamiento Digital de Imagenes
# Ejercicio 3


import numpy as np
from PIL import Image
import math as m


# Cargamos una imagen (Cambiar el nombre del parametro)
imagen =  Image.open('image (1).png').convert('L')
#imagen =  Image.open('image.png').convert('L')
img_array = np.array(imagen)

# Dimensiones de la imagen
alto, ancho = img_array.shape
# El lienzo debe ser el doble de grande de la imagen
alto_dest, ancho_dest = alto *2, ancho*2
# La imagen final
destino =  np.zeros((alto_dest, ancho_dest), dtype=np.uint8)


def afin (x,y, grados, a_min = np.min(img_array), a_max = np.max(img_array)):

    # Obtenemos los valores maximos y minimos de la imagen
    a_l = np.min(img_array)
    a_h = np.max(img_array)

    # Con esto evitamos divisiones entre 0
    if a_h != a_l:
        alpha =  (a_max - a_min) / (a_h - a_l)
    else:
        alpha = 1.0

    # Beta
    beta = a_min - (a_l * alpha)

    theta = m.radians(grados)

    # Matriz afin para traslacion, rotacion e intensidad
    arr = np.array([
        [m.cos(theta), -m.sin(theta), 0, y],
        [m.sin(theta),m.cos(theta), 0, x],
        [0,0,alpha, beta],
        [0,0,0,1]
    ])

    # Obtenemos la inversa para hacer un mapeo hacia atras
    inversa = np.linalg.inv(arr)

    # Obtenemos indices
    y_dest, x_dest = np.indices((alto_dest, ancho_dest))

    # Creamos los vectores que seran asociados a los pixeles mapeados en nuestra imagen transformada
    coords_destino = np.stack([
        x_dest.flatten(),
        y_dest.flatten(),
        np.zeros(ancho_dest * alto_dest), # La intensidad
        np.ones(alto_dest * ancho_dest) # La homogenea
    ])

    # Realizamos el producto de la matriz
    coords_origen = inversa @ coords_destino

    # Redondeamos los valores a numeros enteros, con el fin de evitar decimales
    x_origen = np.round(coords_origen[0, :]).astype(int)
    y_origen = np.round(coords_origen[1, :]).astype(int)

    # Nos ayudara a mantener los pixeles dentro de la region de la imagen transformada.
    mascara_validos = (x_origen >= 0) & (x_origen < ancho) & (y_origen >= 0) & (y_origen < alto)

    # Validamos las coordenadas con la mascara
    x_val = x_origen[mascara_validos]
    y_val = y_origen[mascara_validos]

    # Obtenemos los valores de intensidad originales
    intensidad_original = img_array[y_val, x_val]

    # Organizamos el vector
    vectores = np.stack([
        x_val,
        y_val,
        intensidad_original,
        np.ones(len(x_val))
    ])

    # Realizamos un producto directo con la matriz afin para los valores de intensidad
    v_destino = arr @ vectores

    # Recuperamos unicamente los valores de intensidad obtenidos
    intensidad_aplicada = v_destino[2,:]

    # equivalencia a las condiciones de z <= al y z >= ah
    intensidad_aplicada = np.clip(intensidad_aplicada, a_min, a_max)

    # Generamos un arreglo plano, es decir de una dimension
    img_plano = destino.flatten()

    # Aplicamos la intensidad
    img_plano[mascara_validos] = intensidad_aplicada

    # Volvemos a la matriz
    img_destino = img_plano.reshape((alto_dest, ancho_dest))

    # Guardamos la matriz
    img = Image.fromarray(img_destino, mode='L')
    img.save('imagenAlterada.png')

    return


afin(1000, 200, 270, 0, 100 )


