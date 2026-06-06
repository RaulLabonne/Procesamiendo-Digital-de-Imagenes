import numpy as np
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from scipy.ndimage import label as ndlabel
from skimage import exposure


def slic (input, nsp):
    """
    Algoritmo SLIC modificado para segmentacion de imagenes celulares
    :param input: La imagen a color de entrada.
    :param nsp: Numero de superpixeles
    :return: Imagen de etiquetas
    """

    # Inicializacion de valores
    H,W = input.shape
    nt = H * W
    s = int(np.sqrt(nt/nsp))

    img_mejorada = exposure.equalize_adapthist(input) * 255.0

    medias = initMedias(img_mejorada, s, H, W)

    etiquetas = np.full((H,W), -1, dtype=int)
    distancias = np.full((H,W), np.inf)

    # Algoritmo SLIC Modificado
    tau = 1.0 # Umbral
    error = np.inf # Convergencia
    c = 10.0 # Valor C
    iteracion = 0 # Limite de iteracion

    while error >= tau and iteracion < 10:
        iteracion += 1
        distancias[:] = np.inf  # reseteamos para poder tener margen de mejora por cada cambio de medias
        medias_anteriores = np.copy(medias)

        for i, media in enumerate(medias):

            ## 1: Asignacion de etiquetas
            cx, cy = int(media[1]), int(media[2])

            x_min = max(cx - s, 0)
            x_max = min(cx + s, H)
            y_min = max(cy - s, 0)
            y_max = min(cy + s, W)

            # Iteramos sobre cada pixel de un superpixel
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):

                    # Verificamos que se encuentre dentro del superpixel circular
                    if (x - cx) ** 2 + (y - cy) ** 2 > s ** 2:
                        continue

                    gris = img_mejorada[x, y]
                    pixel = [gris, x, y]

                    D = distancia(pixel, media, s, c)

                    if D < distancias[x, y]:
                        etiquetas[x, y] = i
                        distancias[x, y] = D

        # 2: Actualizacion de representantes
        for i in range(len(medias)):
            mask = (etiquetas == i)

            if np.any(mask):
                mi = np.mean(img_mejorada[mask])

                cx, cy = np.nonzero(mask)
                x_prom = np.mean(cx)
                y_prom = np.mean(cy)

                medias[i] = [mi, x_prom, y_prom]

        # 3: Verificacion de convergencia
        diferencias = medias - medias_anteriores
        error = np.sum(np.linalg.norm(diferencias, axis=1))

    # Post procesamiento
    mask_huerfanos = (etiquetas == -1)
    if np.any(mask_huerfanos):
        # Forzamos los huérfanos temporalmente a la etiqueta 0
        # enforce_connectivity los fusionará con los vecinos correctos
        etiquetas[mask_huerfanos] = 0

    etiquetas = enforce_connectivity(etiquetas, H, W)
    etiquetas = relabel_sequential_manual(etiquetas)

    # Imagen de salida
    output = np.zeros_like(input)

    for i in np.unique(etiquetas):
        mask = (etiquetas == i)
        output[mask] = np.mean(input[mask], axis=0)

    return output, etiquetas



def calcular_gradiente_2d(img_gris):
    """
    Funcion que devuelve la gradiante de una imagen
    :param img_gris:  La region de la imagen gris a sacar la gradiante
    :return: La gradiante
    """
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = Kx.T
    gX = convolve2d(img_gris, Kx, mode='same', boundary='symm')
    gY = convolve2d(img_gris, Ky, mode='same', boundary='symm')
    return gX**2 + gY**2


def initMedias(input, s, H, W):
    """
    Inicializa las medias necesarias para el algoritmo SLIC
    :param input: La imagen mejorada
    :param s: Numero de superpixeles
    :param H: Altura de la imagen
    :param W: Anchura de la imagen
    :return: Arreglo de las medias de cada superpixel
    """
    gradiantes = calcular_gradiente_2d(input)
    medias = []

    # Tomamos el pixel central de cada superpixel
    for cx in range(s // 2, H, s):
        for cy in range(s // 2, W, s):
            x_min = max(cx - 1, 0)
            x_max = min(cx + 2, H)

            y_min = max(cy - 1, 0)
            y_max = min(cy + 2, W)

            vecindad = gradiantes[x_min: x_max, y_min: y_max]
            min_local = np.unravel_index(np.argmin(vecindad), vecindad.shape)

            mejor_x = x_min + min_local[0]
            mejor_y = y_min + min_local[1]

            gris = input[mejor_x, mejor_y]

            medias.append([gris, mejor_x, mejor_y])
    return np.array(medias)

def distancia(pi, pj, s, m):
    """
    Calcula la distancia entre dos vectores de pixeles
    :param pi: Pixel i
    :param pj: Pixel j
    :param s: numero de superpixeles
    :param m: valor arbitrario
    :return: Distancia entre el pixel i y j
    """
    dg = np.sqrt((pi[0] - pj[0]) ** 2)

    ds = np.sqrt((pi[1] - pj[1]) ** 2 + (pi[2] - pj[2]) ** 2)

    a = s / m

    D = np.sqrt(ds ** 2 + a * (dg ** 2))

    return D


def dibujar_limites(imagen_original, etiquetas, color_borde=[0, 0, 0]):
    """
    Dibuja los contornos de los superpíxeles sobre la imagen original.

    :param imagen_original: La imagen original a color.
    :param etiquetas: La matriz de etiquetas devuelta por SLIC.
    :param color_borde: Color del borde en formato [R, G, B].
    :return: Imagen con los bordes dibujados.
    """
    # Creamos una máscara booleana vacía
    bordes = np.zeros(etiquetas.shape, dtype=bool)

    # Comparamos la matriz de etiquetas consigo misma, pero desplazada 1 píxel.
    # Si la etiqueta cambia respecto al vecino, marcamos True (es un borde).

    bordes[:-1, :] |= (etiquetas[:-1, :] != etiquetas[1:, :])

    bordes[:, :-1] |= (etiquetas[:, :-1] != etiquetas[:, 1:])

    # Hacemos una copia de la imagen para no rayar la original
    imagen_con_bordes = np.copy(imagen_original)

    # Si la máscara es True, pintamos el píxel con el color del borde
    imagen_con_bordes[bordes] = color_borde

    return imagen_con_bordes

def enforce_connectivity(etiquetas, H, W):
    """
    Fuerza la conectividad entre regiones y pixeles huerfanos
    :param etiquetas: Imagen de etiquetas
    :param H: Alturea de la imagen
    :param W: Anchura de la imagen
    :return: Imagen con reduccion de pixeles huerfanos
    """
    nueva_etiqueta = np.copy(etiquetas)
    segment_size = np.bincount(etiquetas.ravel())

    for sp in np.unique(etiquetas):
        mascara = (etiquetas == sp)
        componentes, n = ndlabel(mascara)

        if n <= 1:
            continue

        tamaños = [np.sum(componentes == c) for c in range(1, n + 1)]
        orden = np.argsort(tamaños)[::-1]

        for idx in orden[1:]:
            comp = idx + 1
            region = (componentes == comp)
            coords = np.argwhere(region)

            vecinos = []
            for x, y in coords:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < H and 0 <= ny < W:
                        etq_vecina = nueva_etiqueta[nx, ny]
                        if etq_vecina != sp:
                            vecinos.append(etq_vecina)

            if vecinos:
                label_nueva = max(set(vecinos), key=lambda l: segment_size[l])
                nueva_etiqueta[region] = label_nueva
                segment_size[label_nueva] += np.sum(region)
                segment_size[sp] -= np.sum(region)

    return nueva_etiqueta


def relabel_sequential_manual(etiquetas):
    """
    Reenumera las etiqueta de la imagen
    :param etiquetas: Imagen de etiquetas
    :return: Imagen de etiquetas con renumeracion de las mismas a 0,1,2,... sin huecos.
    """
    unicas = np.unique(etiquetas)
    mapping = {v: i for i, v in enumerate(unicas)}
    resultado = np.copy(etiquetas)
    for v, i in mapping.items():
        resultado[etiquetas == v] = i
    return resultado