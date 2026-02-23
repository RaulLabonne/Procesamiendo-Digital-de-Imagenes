import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math as m

imagen =  Image.open('image.png').convert('L')
img_array = np.array(imagen)
alto, ancho = img_array.shape
alto_dest, ancho_dest = alto *2, ancho*2
destino =  np.zeros((alto_dest, ancho_dest), dtype=np.uint8)

def afin (x,y, grados):
    theta = m.radians(grados)
    arr = np.array([
        [m.cos(theta), -m.sin(theta),y],
        [m.sin(theta),m.cos(theta),x],
        [0,0,1]
    ])

    inversa = np.linalg.inv(arr)

    y_dest, x_dest = np.indices((alto_dest, ancho_dest))

    coords_destino = np.stack([
        x_dest.flatten(),
        y_dest.flatten(),
        np.ones(alto_dest * ancho_dest)
    ])

    coords_origen = inversa @ coords_destino

    x_origen = np.round(coords_origen[0, :]).astype(int)
    y_origen = np.round(coords_origen[1, :]).astype(int)

    # El código corregido:
    mascara_validos = (x_origen >= 0) & (x_origen < ancho) & (y_origen >= 0) & (y_origen < alto)

    img_plano = destino.flatten()

    img_plano[mascara_validos] = img_array[y_origen[mascara_validos], x_origen[mascara_validos]]

    img_destino = img_plano.reshape((alto_dest, ancho_dest))

    img = Image.fromarray(img_destino, mode='L')
    img.save('imagenAlterada.png')

    return

afin(900,500 , -90)


