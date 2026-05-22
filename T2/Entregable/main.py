import numpy as np
from PIL import Image
from scipy.signal import convolve2d

import canny
import frecuencias as frec
import espacial_frecuencial as esp_frec
import fouriere



def main():

    # --------------------- Ejercicio 4 a) --------------------
    ruta = 'imagenes/ejercicio3/ejemplos/'
    archivo = '1.jpg'
    img_array = abrir(ruta, archivo, float)

    # Valores de Sigma para filtro Gaussiano
    sigmas = [1,2, 3]

    thi = 20
    tlo = 4

    # Aplicacion de Canny con diferentes sigmas
    bordes = [canny.canny(img_array, sigma, thi, tlo) for sigma in sigmas]

    imgs_bin, imgs_nms = zip(*bordes)

    rutaBordes = 'imagenes/ejercicio3/resultados/'
    rutaENMS = 'imagenes/ejercicio3/NMS/'
    for i in range(3):
        guardar(rutaBordes, imgs_bin[i], f"bordes{i}", '.png')
        guardar(rutaENMS, imgs_nms[i], f"nms{i}", '.png')

    # --------------------- Ejercicio 4 b) --------------------

    nms_s1 = imgs_nms[0]  # Exacto pero con ruido
    nms_s2 = imgs_nms[1] # Equilibrio
    nms_s3 = imgs_nms[2] # Desplazado pero sin ruido

    umbral = 0.02
    mascara_1 = (nms_s3 > umbral).astype(np.uint8)

    umbral = 0.2
    mascara_2 = (nms_s2 > umbral).astype(np.uint8)

    umbral = 1
    mascara_3 = (nms_s3 > umbral).astype(np.uint8)

    mascara_acumulada = mascara_1 + mascara_2 + mascara_3

    # Recuperamos los bordes mas resaltados de la mascara acumulada
    mascara_final = (mascara_acumulada >= 2.3).astype(int)

    # Realzamos un poco los bordes obtenidos para finalizar el proceso de obtencion de la mascara
    mascara_final = (nms_s1 * 1.15) * mascara_final
    nms_mezclado = nms_s1 * mascara_final

    nms_final = np.clip(nms_mezclado, 0, 255).astype(np.uint8)
    ruta = 'imagenes/ejercicio3/mezcla/'
    guardar(ruta, nms_final, 'nms_mix', '.png')

    ebin_mezclada = np.zeros_like(nms_final)

    # Realizamos el trazado umbralizado del algoritmo de Canny para la obtencion de la imagen binaria
    m,n = nms_final.shape
    for i in range(1,m-1):
        for j in range(1,n-1):
            if nms_final[i,j] >= thi and ebin_mezclada[i,j] == 0:
                canny.trazado_umbralizado(nms_final, ebin_mezclada, i, j, tlo)
    ruta = 'imagenes/ejercicio3/mezcla/'

    guardar(ruta, ebin_mezclada, 'ebin_mix', '.png')

    # --------------------- Ejercicio 5 a) --------------------

    archivo = 'prueba1.jpeg'
    ruta = 'imagenes/ejercicio4/a/'
    img_array = abrir(f"{ruta}ejemplos/", archivo, complex)

    ## Filtro pasa baja, pasa alta y pasa media
    fc = [10, 500, 50]
    w = [60, 950, 20]
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
            filtrado = normalizar(filtrado)
            espectro = normalizar(espectro)

            guardar(f"{ruta}resultados/pasa_{pasa_banda[i]}/", filtrado, f"resultado{filtro}{i}", '.png')
            guardar(f"{ruta}espectros/pasa_{pasa_banda[i]}/", espectro, f"espectro{filtro}{i}", '.png')

    # --------------------- Ejercicio 5 b) --------------------
    ruta = 'imagenes/ejercicio4/b/'
    archivo = 'prueba2.jpeg'
    img_array = abrir(f"{ruta}ejemplos/", archivo, float)

    # Valores para el kernel DoG
    sigma1 = 0.08
    sigma2 = 1
    A,B = 1.0, 1.0

    # Obtencion del kernel y su convolucion a las imagenes originales
    kernel_DoG = esp_frec.DoG(sigma1, sigma2, A, B)
    matriz_DoG = convolve2d(img_array, kernel_DoG, mode='same', boundary='wrap')

    kernel_laplas = esp_frec.laplaciano()
    matriz_Laplace = convolve2d(img_array, kernel_laplas, mode='same', boundary='wrap')

    # Normalizamos para poder tener la imagen original con el filtro aplicado
    matriz_Laplace_norm = np.abs(matriz_Laplace)
    matriz_Laplace_norm = np.clip(matriz_Laplace_norm, 0 ,255).astype(np.uint8)
    matriz_DoG_norm = np.abs(matriz_DoG)
    matriz_DoG_norm = np.clip(matriz_DoG_norm, 0 ,255).astype(np.uint8)

    guardar(f"{ruta}espacial/", matriz_DoG_norm, 'imagenDoG')
    guardar(f"{ruta}espacial/", matriz_Laplace_norm, 'imagenLaplace')

    # Objeto que servira para aplicar las frecuencias
    transformada = frec.frecuencias(img_array)

    # Obtenemos la frecuencia asociada a los kernel de Laplace y DoG
    frec_DoG = esp_frec.frecuencia(kernel_DoG, transformada.P, transformada.Q)
    frec_Laplace = esp_frec.frecuencia(kernel_laplas, transformada.P, transformada.Q)

    # ignoramos fake
    matriz_frec_DoG, fake = transformada.pasa_banda(frec_DoG)
    matriz_frec_Laplace, fake = transformada.pasa_banda(frec_Laplace)

    recorte = 15

    # Obtenemos las regiones centrales para la diferencia numerica
    DoG_espacial_centro = matriz_DoG[recorte:-recorte, recorte:-recorte]
    DoG_frec_centro = matriz_frec_DoG[recorte:-recorte, recorte:-recorte]

    Laplace_espacial_centro = matriz_Laplace[recorte:-recorte, recorte:-recorte]
    Laplace_frec_centro = matriz_frec_Laplace[recorte:-recorte, recorte:-recorte]

    print("Diferencia de Gaussianas (Sin bordes)")
    frobenius(DoG_espacial_centro, DoG_frec_centro)

    print("Diferencia de Laplaces (Sin bordes)")
    frobenius(Laplace_espacial_centro, Laplace_frec_centro)

    # Normalizamos para poder tener imagenes claras con su filtro aplicado
    matriz_frec_DoG = normalizar(matriz_frec_DoG)
    matriz_frec_Laplace = normalizar(matriz_frec_Laplace)

    guardar(f"{ruta}frecuencial/", matriz_frec_DoG, 'imagenDoG')
    guardar(f"{ruta}frecuencial/", matriz_frec_Laplace, 'imagenLaplace' )

    # --------------------- Ejercicio 6  --------------------

    ruta = 'imagenes/ejercicio5/'
    archivos = ['prueba0', 'prueba1']
    imgs_arrays = [abrir(f"{ruta}/ejemplos/", f"{archivo}.png", float) for archivo in archivos]
    i = 0
    for img in imgs_arrays:
        m,n = img.shape
        xv, yv = np.meshgrid(np.arange(n), np.arange(m))
        centrado = (-1.0) ** (xv + yv)

        img = img * centrado
        # Aplicacion de nuestra transformada rapida de fourier
        espectro = fouriere.fourier_transform(img)
        espectro = np.log(np.abs(espectro) + 1)

        # Normalizado para visualizacion del espectro resultante
        min_val = np.min(espectro)
        max_val = np.max(espectro)

        espectro_escalado = (espectro - min_val) / (max_val - min_val)
        espectro_final = (espectro_escalado * 255.0).astype(np.uint8)

        guardar(f"{ruta}frecuencias/", espectro_final, f"{archivos[i]}", '.png')
        i += 1

def frobenius(m1, m2):
    dif = m1 - m2
    norma = np.linalg.norm(dif, 'fro')
    print(f"La diferencia numerica es: {norma}\n")

def abrir(ruta, imagen, type):
    archivos = Image.open(f"{ruta}{imagen}").convert('L')
    return np.array(archivos, dtype = type)

def guardar(ruta, matriz, nombre, tipo = '.jpg'):
    imagen = Image.fromarray(matriz, mode='L')
    imagen.save(f"{ruta}{nombre}{tipo}")

def normalizar(gp):
    m, n = gp.shape
    x, y = np.meshgrid(range(n), range(m))
    gp = gp * ((-1) ** (x+y))

    gp = np.abs(gp)

    gp_min = np.min(gp)
    gp_max = np.max(gp)

    if gp_max - gp_min == 0:
        normalizado = gp
    else:
        normalizado = ((gp - gp_min) / (gp_max - gp_min)) * 255

    # 4. Ahora sí, convertimos a uint8 de forma segura
    return normalizado.astype(np.uint8)

def normalizar_espacial(matriz):
    m_min = np.min(matriz)
    m_max = np.max(matriz)
    return (matriz - m_min) / (m_max - m_min)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


#fourier = fouriere.fourier_transform(img_array)

