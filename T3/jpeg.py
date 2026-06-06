import numpy as np

def init(img):
    """
        Rellena la imagen si es necesario y la divide en bloques de 8x8.
        :param img: Imagen de entrada.
        :returns Una lista de bloques de 8x8 y las nuevas dimensiones por realizar un pading
    """
    alto, ancho = img.shape

    # Calculamos cuántos píxeles faltan para llegar a un múltiplo de 8
    falta_alto = (8 - (alto % 8)) % 8
    falta_ancho = (8 - (ancho % 8)) % 8

    # Aplicamos un padding
    if falta_alto > 0 or falta_ancho > 0:
        imagen_pad = np.pad(img,
                            pad_width=((0, falta_alto), (0, falta_ancho)),
                            mode='edge')
    else:
        imagen_pad = img  # Si ya es multiplo de 8

    # Extraemos los bloques de 8x8
    nuevo_alto, nuevo_ancho = imagen_pad.shape
    bloques_8x8 = []

    # Recorremos la imagen dando saltos de 8 en 8
    for i in range(0, nuevo_alto, 8):
        for j in range(0, nuevo_ancho, 8):
            # Recortamos la submatriz de 8x8
            bloque = imagen_pad[i:i + 8, j:j + 8]
            bloques_8x8.append(bloque)

    return bloques_8x8, nuevo_alto, nuevo_ancho


def calcular_dct_8x8(bloque_g):
    """
    Calcula la Transformada Discreta de Coseno de un bloque 8x8
    :param bloque_g: Bloque 8x8
    :return El bloque con DCT aplicado
    """
    G = np.zeros((8, 8), dtype=np.float64)

    # Creamos una mallas de índices x, y para la vectorización
    x, y = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')

    for u in range(8):
        for v in range(8):
            # Funciones de normalización C(u) y C(v)
            Cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
            Cv = 1.0 / np.sqrt(2) if v == 0 else 1.0

            # Calculamos los términos de los cosenos de forma matricial
            termino_x = np.cos(((2 * x + 1) * u * np.pi) / 16.0)
            termino_y = np.cos(((2 * y + 1) * v * np.pi) / 16.0)

            # Sumatoria doble: multiplicamos el bloque por ambos términos y sumamos
            suma = np.sum(bloque_g * termino_x * termino_y)

            # Aplicamos la fórmula completa
            G[u, v] = 0.25 * Cu * Cv * suma

    return G

def cuantizacion(G: np.ndarray):
    """
    Aplica cuantizacion usando una matriz tipica para JPEG
    :param G: Bloque DCT de 8x8
    :return: El bloque cuantizado
    """
    # Matriz de cuantizacion tipica para JPEG
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    G_q = np.round(np.divide(G, Q))
    return G_q


def obtener_ruta_zigzag(n, m):
    """
        Genera la lista de coordenadas (fila, columna) en el orden zigzag
        :param n: Numero de fila
        :param m: Numero de columna
        :return Las coordenadas para mapear una matriz a un arreglo siguiendo el orden zigzag
    """
    coordenadas = []
    # La suma de los índices (i + j) va desde 0 hasta (n + m - 2)
    for s in range(n + m - 1):
        if s % 2 == 0:
            # Diagonal PAR: Subiendo (la fila 'i' disminuye)
            for i in range(min(n - 1, s), max(0, s - m + 1) - 1, -1):
                j = s - i
                coordenadas.append((i, j))
        else:
            # Diagonal IMPAR: Bajando (la fila 'i' aumenta)
            for i in range(max(0, s - m + 1), min(n - 1, s) + 1):
                j = s - i
                coordenadas.append((i, j))
    return coordenadas

def codificacion(G_q: np.ndarray):
    """
    Codifica los bloques siguiendo un patron en zigzag
    :param G_q: Bloque 8x8 cuantizado
    :returns: El arreglo mapeado y codificado(AC), las dimensiones del bloque (8,8) y el DC del bloque
    """
    n, m = G_q.shape
    ruta = obtener_ruta_zigzag(n, m)

    # Construimos el vector leyendo la matriz en el orden de la ruta
    secuencia_completa = [G_q[i, j] for i, j in ruta]

    # Separamos el DC
    dc = secuencia_completa[0]

    # AC
    secuencia_zigzag = secuencia_completa[1:]

    acc_zeros = 0
    codificado = []

    for num in secuencia_zigzag:
        if num == 0:
            acc_zeros += 1
            continue
        elif acc_zeros > 0:
            codificado.append(f"{acc_zeros}")
            acc_zeros = 0
        codificado.append(num)

    # En caso de que termine en ceros
    if acc_zeros > 0:
        codificado.append(f"{acc_zeros}")
    return codificado, n, m, dc


def decodificacion(zigzag, dc):
    """
    Decodifica el arreglo
    :param zigzag: El arreglo a decodificar
    :param dc: El componente DC que ignoramos de dicho bloque
    :return: El arreglo decodificado correspondiente a un bloque
    """
    decodificado = []

    for elemento in zigzag:
        if type(elemento) == str:
            n = int(elemento)
            for i in range(n):
                decodificado.append(0)
        else:
            decodificado.append(elemento)
    return [dc] + decodificado

def zigzag_inverso(zigzag, n, m):
    """
    Obtiene el bloque de 8x8 a partir del arreglo, recuperandolo considerando la forma zigzag
    :param zigzag: El arreglo
    :param n: Alto del bloque
    :param m: Ancho del bloque
    :return: Un bloque de 8x8 a partir del arreglo
    """
    G_q = np.zeros((n, m), dtype=type(zigzag[0]))
    ruta = obtener_ruta_zigzag(n, m)

    # Colocamos cada elemento de la secuencia en su posición correspondiente
    for k, (i, j) in enumerate(ruta):
        G_q[i, j] = zigzag[k]

    return G_q

def cuantizacion_inv(G_q: np.ndarray):
    """
    Aplica la cuantizacion inversa usando la matriz tipica de JPEG
    :param G_q: EL bloque de 8x8
    :return: El bloque sin cuantizar
    """
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    return G_q * Q

def calcular_idct_8x8(G_q):
    """
    Calcula la Transformada Inversa Discreta de Coseno de un bloque de frecuencias 8x8
    :param G_q: EL bloque de 8x8
    :return El bloque decodificado final
    """
    G = np.zeros((8, 8), dtype=np.float64)

    u, v = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')

    Cu = np.where(u == 0, 1.0 / np.sqrt(2), 1.0)
    Cv = np.where(v == 0, 1.0 / np.sqrt(2), 1.0)

    # Recorremos las filas y columnas del bloque de imagen (x, y)
    for x in range(8):
        for y in range(8):
            # Calculamos los términos de los cosenos usando la malla u, v
            termino_x = np.cos(((2 * x + 1) * u * np.pi) / 16.0)
            termino_y = np.cos(((2 * y + 1) * v * np.pi) / 16.0)

            # Sumatoria doble: multiplicamos las frecuencias por sus coeficientes
            # y cosenos, y luego sumamos
            suma = np.sum(G_q * Cu * Cv * termino_x * termino_y)

            # Aplicamos la fórmula completa para el píxel actual
            G[x, y] = 0.25 * suma

    return G

def imagen_jpg(bloques, n,m, n_new, m_new):
    """
    Reconstruye la imagen a partir de un listado de bloques
    :param bloques: La lista de bloques de 8x8
    :param n: La longitud original de la imagen
    :param m: La anchura original de la imagen
    :param n_new: La longitud nueva de la imagen
    :param m_new: La anchura nueva de la imagen
    :return: La imagen descomprimida por JPEG
    """
    imagen_reconstruida = np.zeros((n_new, m_new), dtype=np.float64)

    # Pegamos los bloques en el lienzo
    indice_bloque = 0
    for i in range(0, n_new, 8):
        for j in range(0, m_new, 8):
            # Asignamos el bloque actual a la porción correspondiente de la matriz
            imagen_reconstruida[i:i + 8, j:j + 8] = bloques[indice_bloque]
            indice_bloque += 1

    # Recortamos el padding para volver al tamaño original
    imagen_final = imagen_reconstruida[0:n, 0:m]

    imagen_final = np.clip(imagen_final, 0, 255)
    imagen_final = np.round(imagen_final).astype(np.uint8)

    return imagen_final