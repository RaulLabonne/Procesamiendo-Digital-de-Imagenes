import slic_plus
import slic
import k_media
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from PIL import Image
from medsegbench import CellnucleiMSBench

def main():

    # Base de datos de imagenes celulares
    data_set = CellnucleiMSBench(split='train', download='True')
    nsp = 750

    prom_dice, prom_dice_slic = 0, 0
    prom_iou, prom_iou_slic = 0, 0

    for i in range(10):
        img_data, mask_data = data_set[i]

        img_np = np.array(img_data)
        mask_np = np.array(mask_data)

        # Slic modificado
        gray = rgb2gray(img_np)
        # Slic base
        lab = rgb2lab(img_np)

        # Slic modificado
        salida, bordes = slic_plus.slic(gray, nsp)
        salida = (salida * 255).astype(np.uint8)
        # Slic base
        s_slic, b_slic = slic.slic(lab, nsp)
        s_slic = lab2rgb(s_slic)
        s_slic = (s_slic * 255).astype(np.uint8)

        # k-medias para obtener las mascaras

        Alto, Ancho = salida.shape
        AS, AnS, CS = s_slic.shape

        Z = salida.reshape((-1, 2))
        Z_slic = s_slic.reshape((-1, 3))

        k = 2
        etiquetas, M = k_media.k_medias(Z, k)
        e1, M1 = k_media.k_medias(Z_slic, k)

        #  Creamos nuevas matrices de centroides forzadas a blanco y negro
        M_binario = np.zeros_like(M)
        M1_binario = np.zeros_like(M1)

        # Identificamos cuál grupo es el más claro y cuál el más oscuro
        # Calculamos el brillo promedio de cada centroide (sumando sus canales RGB)
        brillo_M = np.sum(M, axis=1)
        brillo_M1 = np.sum(M1, axis=1)

        # Asignamos [255, 255, 255] al grupo con mayor brillo
        M_binario[np.argmax(brillo_M)] = [255, 255]
        M1_binario[np.argmax(brillo_M1)] = [255, 255, 255]

        imagen_coloreada = M_binario[etiquetas]
        i_test = M1_binario[e1]

        # Le regresamos su forma original de imagen
        imagen_final = imagen_coloreada.reshape((Alto, Ancho))
        i_test_final = i_test.reshape((AS,AnS,CS))

        mascara_o = Image.open(f'./images/mask/{i}.png').convert('RGB')
        arr_original = np.array(mascara_o)

        # La imagen 2 al ser blanco el fondo, lo invertimos para tener una mascara similar a la original
        if i == 2:
            imagen_final = inversa(imagen_final)
            i_test_final = inversa(i_test_final)
        # Calculamos el coeficiente DICE y IoU
        dice, iou = evaluar_segmentacion(imagen_final, arr_original)
        dice_slic, iou_slic = evaluar_segmentacion(i_test_final, arr_original)
        prom_dice += dice
        prom_iou += iou
        prom_dice_slic += dice_slic
        prom_iou_slic += iou_slic
        print(f"---Imagen: {i}---")
        print("-Slic modificado-")
        print(f"Dice: {dice}\nIoU: {iou}")
        print("-Slic normal-")
        print(f"DICE: {dice_slic}\nIoU: {iou_slic}")

        # Aseguramos que los valores sean correctos para una imagen
        imagen_final = np.clip(imagen_final, 0, 255).astype(np.uint8)
        i_test_final = np.clip(i_test_final, 0, 255).astype(np.uint8)


        # Creamos las imagenes con bordes
        b_slic = slic.dibujar_limites(img_np, b_slic, [255,255,255])
        img_bordes = slic_plus.dibujar_limites(img_np, bordes, [255,255,255])

        # Imagenes a guardar: slic, islic, mascara original, mascara, imagen original
        imagen = Image.fromarray(salida)
        img_bordes = Image.fromarray(img_bordes)
        img_mask = Image.fromarray(mask_np)
        img_or = Image.fromarray(img_np)

        img_slic = Image.fromarray(s_slic)
        brd_slic = Image.fromarray(b_slic)

        img_final = Image.fromarray(imagen_final)
        comparar = Image.fromarray(i_test_final)
        img_final.save(f'./images/mask_final/{i}.png')
        comparar.save(f'./images/mask_final/{i}compare.png')


        img_slic.save(f"./slic_img/segmentado/{i}.png")
        brd_slic.save(f"./slic_img/borde/{i}.png")

        imagen.save(f"./images/resultados/{i}.png")
        img_bordes.save(f"./images/bordes/{i}.png")
        img_mask.save(f"./images/mask/{i}.png")
        img_or.save(f"./images/original/{i}.png")

    print(f"Promedio DICE: {prom_dice/10}")
    print(f"Promedio IoU: {prom_iou/10}")
    print("-VS-")
    print(f"Promedio DICE Slic: {prom_dice_slic/10}")
    print(f"Promedio IoU Slic: {prom_iou_slic/10}")
    return

def evaluar_segmentacion(mascara_predicha, mascara_real):
    """
    Calcula el coeficiente DICE y el IoU entre dos imágenes binarias.

    :param mascara_predicha: La imagen binaria obtenida de tu K-medias (predicción).
    :param mascara_real: La imagen binaria original (Ground Truth).
    :return: Tupla con (Coeficiente Dice, IoU).
    """
    if mascara_predicha.ndim == 3:
        mascara_predicha = mascara_predicha[:, :, 0]

    if mascara_real.ndim == 3:
        mascara_real = mascara_real[:, :, 0]

    # Asegurarnos de que ambas matrices sean estrictamente booleanas (True/False o 1/0)
    pred = mascara_predicha > 127  # Ajusta el umbral si es necesario
    real = mascara_real > 127

    # Calcular la intersección (Píxeles donde ambas máscaras son 'True')
    interseccion = np.logical_and(pred, real).sum()

    # Calcular la Unión para el IoU (Píxeles donde al menos una es 'True')
    union = np.logical_or(pred, real).sum()

    # Calcular la suma total de píxeles activos para el Dice
    suma_elementos = pred.sum() + real.sum()

    # Prevenir divisiones por cero en caso de que ambas imágenes sean completamente negras
    if union == 0:
        iou = 1.0 if suma_elementos == 0 else 0.0
    else:
        iou = interseccion / union

    if suma_elementos == 0:
        dice = 1.0
    else:
        dice = (2.0 * interseccion) / suma_elementos

    return dice, iou

def inversa(imagen2):
    """
    Funcion exclusiva de imagen 2
    :param imagen2: La imagen 2
    :return: Imagen binaria invertida
    """
    return 255 - imagen2


if __name__ == '__main__':
    main()
