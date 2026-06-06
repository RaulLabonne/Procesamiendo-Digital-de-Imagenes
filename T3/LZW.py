import numpy as np

import numpy as np


def lzw_encode(image: np.ndarray):
    """
    Codifica una imagen 2D usando el algoritmo LZW.
    Devuelve un arreglo de NumPy optimizado en espacio.
    """
    # Aplanamos la imagen para procesarla como una secuencia de bytes
    flat_data = image.flatten()

    # Inicializamos el diccionario con los primeros 256 caracteres (bytes)
    dict_size = 256
    dictionary = {(i,): i for i in range(dict_size)}

    w = ()
    compressed_data = []

    for symbol in flat_data:
        wc = w + (symbol,)
        if wc in dictionary:
            w = wc
        else:
            compressed_data.append(dictionary[w])
            # Agregamos la nueva secuencia detectada al diccionario
            dictionary[wc] = dict_size
            dict_size += 1
            w = (symbol,)

    if w:
        compressed_data.append(dictionary[w])

    # Guardamos el código optimizando el tipo de dato (sin desperdiciar espacio)
    max_val = max(compressed_data) if compressed_data else 0
    if max_val < 256:
        dtype = np.uint8
    elif max_val < 65536:
        dtype = np.uint16
    else:
        dtype = np.uint32

    return np.array(compressed_data, dtype=dtype)


def lzw_decode(compressed_arr: np.ndarray, original_shape: tuple):
    """
    Decodifica el arreglo LZW y reconstruye la imagen original.
    """
    dict_size = 256
    # El diccionario inverso mapea códigos enteros a tuplas de bytes
    dictionary = {i: (i,) for i in range(dict_size)}

    compressed_list = compressed_arr.tolist()
    if not compressed_list:
        return np.zeros(original_shape, dtype=np.uint8)

    w = dictionary[compressed_list[0]]
    result = list(w)

    for code in compressed_list[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = w + (w[0],)
        else:
            raise ValueError(f"Código de compresión corrupto: {code}")

        result.extend(entry)
        # Reconstruimos el diccionario dinámicamente sobre la marcha
        dictionary[dict_size] = w + (entry[0],)
        dict_size += 1
        w = entry

    # Reestructuramos el arreglo unidimensional a la forma de la imagen original
    return np.array(result, dtype=np.uint8).reshape(original_shape)