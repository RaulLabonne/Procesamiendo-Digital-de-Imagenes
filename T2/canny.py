import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
from numpy import dtype


#from scipy.signal import convolve2d

def aplicar_padding(imagen, alto_pad, ancho_pad, valor_relleno=0):
    alto_img, ancho_img = imagen.shape
    nueva_altura = alto_img + 2 * alto_pad
    nuevo_ancho = ancho_img + 2 * ancho_pad
    imagen_con_padding = np.full((nueva_altura, nuevo_ancho), valor_relleno, dtype=imagen.dtype)
    imagen_con_padding[alto_pad: alto_pad + alto_img, ancho_pad: ancho_pad + ancho_img] = imagen
    return imagen_con_padding


def convolucion_2d_manual(imagen, kernel, orientacion=2, gradiante = False):
    match orientacion:
        case 0:
            kernel = kernel.reshape(1,-1)
        case 1:
            kernel = kernel.reshape(-1,1)

    alto_k, ancho_k = kernel.shape
    pad_alto = alto_k // 2
    pad_ancho = ancho_k // 2

    imagen_con_padding = aplicar_padding(imagen, pad_alto, pad_ancho)
    alto_img, ancho_img = imagen.shape
    imagen_salida = np.zeros_like(imagen, dtype=np.float32)

    kernel_volteado = np.flipud(np.fliplr(kernel))

    for y in range(alto_img):
        for x in range(ancho_img):
            vecindario = imagen_con_padding[y: y + alto_k, x: x + ancho_k]
            imagen_salida[y, x] = np.sum(vecindario * kernel_volteado)
    if not gradiante:
        imagen_salida = np.clip(imagen_salida, 0, 255).astype(np.uint8)
    return imagen_salida


def gauss(sigma, k = 3):
    radio = int(sigma * k)

    ax = np.arange(-radio, radio + 1)

    kernel = np.exp(- (ax ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

def canny(imagen, sigma, thi, tlo):
    # Suavizado con Gaussiana de escala sigma
    k_gauss = gauss(sigma)
    img_gauss = convolucion_2d_manual(imagen, k_gauss, 0)
    img_gauss = convolucion_2d_manual(img_gauss, k_gauss, 1)


    k_gradiante = np.array([-0.5, 0, 0.5])
    gradX = convolucion_2d_manual(img_gauss, k_gradiante, 0, True)
    gradY = convolucion_2d_manual(img_gauss, k_gradiante, 1, True)
    m,n = imagen.shape

    emag = np.zeros_like(imagen, dtype=np.float32)
    enms = np.zeros_like(imagen, dtype=np.float32)
    ebin = np.zeros_like(imagen, dtype=np.uint8)

    emag = np.sqrt(gradX**2 + gradY**2)

    emag_visual = np.clip(emag, 0, 255).astype(np.uint8)


    for i in range(1,m-1):
        for j in range(1,n-1):
            dx = gradX[i,j]
            dy = gradY[i,j]
            s = orientacion(dx,dy)
            if maximo_local(emag, i,j,s, tlo):
                enms[i,j] = emag[i,j]

    enms_visual = np.clip(enms, 0, 255).astype(np.uint8)

    for i in range(1,m-1):
        for j in range(1,n-1):
            if enms[i,j] >= thi and ebin[i,j] == 0:
                trazado_umbralizado(enms, ebin, i, j, tlo)
    return ebin, enms_visual

def orientacion(dx, dy):
    rotacion = np.array([
                        [np.cos(np.pi/8), -np.sin(np.pi/8)],
                        [np.sin(np.pi/8), np.cos(np.pi/8)]
                        ])

    vector = np.array([
                        [dx],
                        [dy]
                        ])

    vector_rotado = rotacion @ vector

    if vector_rotado[1][0] < 0:
        vector_rotado[0][0] = -vector_rotado[0][0]
        vector_rotado[1][0] = -vector_rotado[1][0]

    dx1 = vector_rotado[0][0]
    dy1 = vector_rotado[1][0]

    s = 0

    if dx1 >= 0 and dx1 >= dy1:
        s = 0
    elif dy1 >= 0 and dy1 > dx1:
        s = 1
    elif dx1 < 0 and -dx1 >= dy1:
        s = 2
    elif dy1 < 0 and -dy1 > dx1:
        s = 3
    return s


def maximo_local(emag, v, u, s, tlo):
    mc = emag[v,u]

    if mc < tlo:
        return False
    else:
        ml = 0
        mR = 0
        match s:
            case 0:
                ml = emag[v,u-1]
                mR = emag[v,u+1]
            case 1:
                ml = emag[v-1, u - 1]
                mR = emag[v+1, u+1]
            case 2:
                ml = emag[v-1, u]
                mR = emag[v+1, u]
            case 3:
                ml = emag[v-1, u + 1]
                mR = emag[v+1, u - 1]
        return ml <= mc and mc >= mR

def trazado_umbralizado(enms, ebin, u0, v0, tlo):
    m,n = enms.shape
    ebin[u0,v0] = 255
    uL = max(u0 - 1, 0)
    uR = min(u0 + 1, m - 1)
    vT = max(v0 - 1, 0)
    vB = min(v0 + 1, n-1)

    for u in range(uL, uR+1):
        for v in range(vT, vB+1):
            if enms[u,v] >= tlo and ebin[u,v] == 0:
                trazado_umbralizado(enms, ebin, u, v, tlo)