import slic
import jpeg
import LZW
import sys
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from PIL import Image

def main():

    ## Imagenes a usar
    slic_image1 = Image.open("./assets/SLIC/airplane.bmp")
    slic_image2 = Image.open("./assets/SLIC/goldhill.bmp")
    slic_image3 = Image.open("./assets/SLIC/lena.png")

    slic_images = [slic_image1, slic_image2, slic_image3]

    jpeg_image1 = Image.open("./assets/JPEG/girlface.bmp")
    jpeg_image2 = Image.open("./assets/JPEG/man.bmp")
    jpeg_image3 = Image.open("./assets/JPEG/zelda.bmp")

    jpeg_images = [jpeg_image1, jpeg_image2, jpeg_image3]

    lzw_image1 = Image.open("./assets/LZW/boats.bmp")  # Imagen natural
    lzw_image2 = Image.open("./assets/LZW/test1.gif").convert("L")  #Imagen con patrones
    lzw_image3 = Image.open("./assets/LZW/test2.gif").convert("L")  # Imagen con patrones

    lzw_images = [lzw_image1, lzw_image2, lzw_image3]


    ## Ejercicio 1: SLIC

    nsp = 128    # Numero de superpixeles
    i = 1

    for image in slic_images:
        array = np.array(image)
        lab = rgb2lab(array)
        salida, bordes = slic.slic(lab, nsp)
        salida = lab2rgb(salida)

        salida = (salida * 255).astype(np.uint8)

        img_bordes = slic.dibujar_limites(array, bordes, [255,255,255])

        imagen = Image.fromarray(salida)
        img = Image.fromarray(img_bordes)
        imagen.save(f"./assets/SLIC/result/{i}_slic.jpg")
        img.save(f"./assets/SLIC/result/{i}_borde.jpg")
        i += 1

    # Ejercicio 2: JPG

    i = 1

    for image in jpeg_images:

        array = np.array(image)

        N,M = array.shape

        bloques, x,y = jpeg.init(array)

        bloques_dct = [jpeg.calcular_dct_8x8(bloque) for bloque in bloques]

        bloques_cuantizados = [jpeg.cuantizacion(bloque) for bloque in bloques_dct]

        datos_comprimidos = [jpeg.codificacion(bloque) for bloque in bloques_cuantizados]

        ## -- ##
        bytes_original = array.size

        # Comprimido

        simbolos_comprimidos = 0
        for ac_lista, n,m, dc in datos_comprimidos:
            simbolos_comprimidos += 1
            simbolos_comprimidos += len(ac_lista)

        print("-"*25)
        print(f"Resultados de compresion JPEG para imagen {i}")
        print(f"\nBytes de imagen original: {bytes_original:,} bytes")
        print(f"Simbolos compresion: {simbolos_comprimidos:,} unidades")
        print(f"Reduccion de informacion: {100 * (1-simbolos_comprimidos/bytes_original):.2f}% ")

        ## -- ##

        bloques_recuperados = []

        for ac_lista, n,m, dc in datos_comprimidos:
            zigzag = jpeg.decodificacion(ac_lista, dc)
            G_q = jpeg.zigzag_inverso(zigzag,8,8)
            G_q = jpeg.cuantizacion_inv(G_q)
            Gq = jpeg.calcular_idct_8x8(G_q)
            bloques_recuperados.append(Gq)

        ## -- ##

        imagen_decodificada = jpeg.imagen_jpg(bloques_recuperados, N,M ,x,y)

        imagen_jpeg = Image.fromarray(imagen_decodificada)

        imagen_jpeg.save(f"./assets/JPEG/result/result_{i}.png")

        i += 1

    # Ejercicio 3: LZW

    i = 1

    for image in lzw_images:

        array = np.array(image)

        # 2. Codificación
        img_encoded = LZW.lzw_encode(array)

        # 3. Decodificación
        img_decoded = LZW.lzw_decode(img_encoded, array.shape)

        # c) Medición de la diferencia usando la Norma de Frobenius
        # Convertimos a float para evitar desbordamientos en la resta de enteros
        diff_norm = np.linalg.norm(array.astype(float) - img_decoded.astype(float), ord='fro')

        # Cálculo de tamaños en memoria
        bytes_original = array.nbytes
        bytes_compressed = img_encoded.nbytes
        compression_ratio = bytes_original / bytes_compressed

        # Despliegue de resultados
        print("-"*10)
        print("Resultados de compresion LZW")
        print(f"Dimensiones de la imagen: {array.shape}")
        print(f"Tamaño de la imagen original: {bytes_original} bytes")
        print(f"Tamaño del arreglo codificado: {bytes_compressed} bytes ({img_encoded.dtype})")
        print(f"Relación de compresión: {compression_ratio:.2f}x menos espacio")
        print(f"Norma de Frobenius de la diferencia: {diff_norm}")


    return

if __name__ == '__main__':
    main()
