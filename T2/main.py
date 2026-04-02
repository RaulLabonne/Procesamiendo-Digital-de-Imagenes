import numpy as np
from PIL import Image
import canny
import frecuencias as frec
import espacial_frecuencial as esp_frec
import fouriere



def main():

    # Ejercicio 4
    # Cargamos una imagen (Cambiar el nombre del parametro)
    imagen = Image.open('imagenes/ejercicio4/ejemplos/1.jpg').convert('L')
    img_array = np.array(imagen)

    sigmas = [1,2, 3]
    thi = 20
    tlo = 4

    bordes = [canny.canny(img_array, sigma, thi, tlo) for sigma in sigmas]

    imgs_bin, imgs_nms = zip(*bordes)

    # Guardamos la matriz
    img_bin = [Image.fromarray(final, mode='L') for final in imgs_bin]
    img_nms = [Image.fromarray(final, mode='L') for final in imgs_nms]
    for i in range(3):
        img_bin[i].save(f"imagenes/ejercicio4/resultados/bordes{i}.png")
        img_nms[i].save(f"imagenes/ejercicio4/NMS/nms{i}.png")

    ## Ejercicio 4 b)
    ruta = 'imagenes/ejercicio4/NMS/'
    imagenes_canny = [Image.open(f"{ruta}nms{i}.png") for i in range(3)]
    imagenes_canny = [np.array(i).astype(float) for i in imagenes_canny]

    nms_s1 = imagenes_canny[0]  # Exacto pero con ruido
    nms_s2 = imagenes_canny[1] # Equilibrio
    nms_s3 = imagenes_canny[2] # Desplazado pero sin ruido

    umbral = 0.02

    mascara_1 = (nms_s3 > umbral).astype(np.uint8)

    umbral = 0.2
    mascara_2 = (nms_s2 > umbral).astype(np.uint8)

    umbral = 1
    mascara_3 = (nms_s3 > umbral).astype(np.uint8)

    mascara_acumulada = mascara_1 + mascara_2 + mascara_3

    mascara_final = (mascara_acumulada >= 2.3).astype(int)

    mascara_final = (nms_s1 * 1.15) * mascara_final


    nms_mezclado = nms_s1 * mascara_final

    nms_final = np.clip(nms_mezclado, 0, 255).astype(np.uint8)
    ruta = 'imagenes/ejercicio4/mezcla/'
    img_nms_mix = Image.fromarray(nms_final, mode='L')
    img_nms_mix.save(f"{ruta}nms_mix.png")

    ebin_mezclada = np.zeros_like(nms_final)

    m,n = nms_final.shape
    for i in range(1,m-1):
        for j in range(1,n-1):
            if nms_final[i,j] >= thi and ebin_mezclada[i,j] == 0:
                canny.trazado_umbralizado(nms_final, ebin_mezclada, i, j, tlo)
    ruta = 'imagenes/ejercicio4/mezcla/'
    img_ebin_mix = Image.fromarray(ebin_mezclada, mode='L')
    img_ebin_mix.save(f"{ruta}ebin_mix.png")
    ## Ejercicio 5 a)

    pruebas = 'prueba1.jpeg'
    ruta = 'imagenes/ejercicio5/a/'
    imagen = Image.open(f"{ruta}ejemplos/{pruebas}").convert('L')

    img_array = np.array(imagen, dtype=complex)

    ## Filtro pasa baja, pasa alta y pasa media
    fc = [0, 500, 60]
    w = [60, 950, 40]
    n = 3
    pasa_banda = ['alta', 'baja', 'media']
    filtros = ['Butter', 'Ideal', 'Gauss']

    frecuencias = frec.frecuencias(img_array)

    for i in range(len(pasa_banda)):

        img_filtrada = None

        for filtro in filtros:
            match filtro:
                case 'Butter':
                    img_filtrada = frecuencias.pasa_banda_Butter(w[i], n, fc[i])
                case 'Ideal':
                    img_filtrada = frecuencias.pasa_banda_ideal(w[i], fc[i])
                case 'Gauss':
                    img_filtrada = frecuencias.pasa_banda_Gauss(w[i], fc[i])
                case _:
                    # nunca pasa
                    return

            filtrado, espectro = frecuencias.pasa_banda(img_filtrada)
            filtrado = frecuencias.normalizar(filtrado)

            imgfiltrado = Image.fromarray(filtrado, mode='L')
            imgfiltrado.save(f"{ruta}resultados/pasa_{pasa_banda[i]}/resultado{filtro}{i}.png")
            img_espectro = Image.fromarray(espectro, mode='L')
            img_espectro.save(f"{ruta}espectros/pasa_{pasa_banda[i]}/espectro{filtro}{i}.png")

    # Ejercicio 5 b)
    ruta = 'imagenes/ejercicio5/b/'
    imagen = Image.open(f"{ruta}ejemplos/prueba2.jpeg").convert('L')
    img_array = np.array(imagen,dtype=float)

    sigma1 = 1.00
    sigma2 = 1.50
    kernel_DoG = esp_frec.DoG(sigma1, sigma2)
    matriz_DoG = canny.convolucion_2d_manual(img_array, kernel_DoG)
    kernel_laplas = esp_frec.laplaciano()
    matriz_Laplace = canny.convolucion_2d_manual(img_array, kernel_laplas)

    imagen_DoG = Image.fromarray(matriz_DoG, mode='L')
    imagen_Laplace = Image.fromarray(matriz_Laplace, mode='L')
    imagen_DoG.save(f"{ruta}espacial/imagenDoG.jpg")
    imagen_Laplace.save(f"{ruta}espacial/imagenLaplace.jpg")

    M,N = img_array.shape
    frec_DoG = esp_frec.frecuencia(kernel_DoG, M, N)
    frec_Laplace = esp_frec.frecuencia(kernel_laplas, M, N)

    frecuencias = frec.frecuencias(img_array)

    matriz_frec_DoG, fake = frecuencias.pasa_banda(frec_DoG)

    matriz_frec_Laplace, fake = frecuencias.pasa_banda(frec_Laplace)

    recorte = 15

    # Obtenemos las regiones centrales
    DoG_espacial_centro = matriz_DoG[recorte:-recorte, recorte:-recorte]
    DoG_frec_centro = matriz_frec_DoG[recorte:-recorte, recorte:-recorte]

    Laplace_espacial_centro = matriz_Laplace[recorte:-recorte, recorte:-recorte]
    Laplace_frec_centro = matriz_frec_Laplace[recorte:-recorte, recorte:-recorte]

    print("Diferencia de Gaussianas (Sin bordes)")
    frobenius(DoG_espacial_centro, DoG_frec_centro)

    print("Diferencia de Laplaces (Sin bordes)")
    frobenius(Laplace_espacial_centro, Laplace_frec_centro)

    matriz_frec_DoG = frecuencias.normalizar(matriz_frec_DoG)
    matriz_frec_Laplace = frecuencias.normalizar(matriz_frec_Laplace)

    imagen_DoG = Image.fromarray(matriz_frec_DoG, mode='L')
    imagen_Laplace = Image.fromarray(matriz_frec_Laplace, mode='L')
    imagen_DoG.save(f"{ruta}frecuencial/imagenDoG.jpg")
    imagen_Laplace.save(f"{ruta}frecuencial/imagenLaplace.jpg")



    # Ejercicio 6

    ruta = 'imagenes/ejercicio6/'
    archivos = ['prueba0.png', 'prueba1.png']
    imagenes = [Image.open(f"{ruta}/ejemplos/{i}").convert('L') for i in archivos]
    imgs_arrays = [np.array(imagen, dtype=float) for imagen in imagenes]
    i = 0
    for img in imgs_arrays:
        espectro = fouriere.fourier_transform(img)
        espectro = np.log(np.abs(espectro) + 1)

        min_val = np.min(espectro)
        max_val = np.max(espectro)

        espectro_escalado = (espectro - min_val) / (max_val - min_val)

        espectro_final = (espectro_escalado * 255.0).astype(np.uint8)

        imagen_espectro = Image.fromarray(espectro_final, mode='L')
        imagen_espectro.save(f"{ruta}frecuencias/{archivos[i]}")
        i += 1

def frobenius(m1, m2):
    dif = m1 - m2
    norma = np.linalg.norm(dif, 'fro')
    print(f"La diferencia numerica es: {norma}\n")

def abrir(ruta, imagen, type=float):
    archivos = Image.open(f"{ruta}{imagen}").convert('L')
    return np.array(archivos, dtype = type)

def guardar(ruta, matriz, nombre):
    imagen = Image.fromarray(matriz, mode='L')
    imagen.save(f"{ruta}{nombre}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


#fourier = fouriere.fourier_transform(img_array)

