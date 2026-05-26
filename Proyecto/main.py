import slic_plus
import slic
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from PIL import Image
from medsegbench import CellnucleiMSBench

def main():
    #image = Image.open("./assets/lena_std.jpg")
    #image = Image.open('./assets/1_22_s.bmp')
    #image = Image.open('./assets/3.png')
    #array = np.array(image)
    #array = np.full_like(array1, 255)

    data_set = CellnucleiMSBench(split='train', download='True')

    img_grises = []
    img_binaias=[]
    nsp = 800

    for i in range(10):
        img_data, mask_data = data_set[i]


        img_np = np.array(img_data)
        mask_np = np.array(mask_data)


        gray = rgb2gray(img_np)
        lab = rgb2lab(img_np)
        s_slic, b_slic = slic.slic(lab, nsp)
        salida, bordes = slic_plus.slic(gray, nsp)
        s_slic = (s_slic * 255).astype(np.uint8)
        salida = (salida * 255).astype(np.uint8)

        b_slic = slic.dibujar_limites(img_np, b_slic, [255,255,255])
        img_bordes = slic_plus.dibujar_limites(img_np, bordes, [255,255,255])

        imagen = Image.fromarray(salida)
        img_bordes = Image.fromarray(img_bordes)
        img_mask = Image.fromarray(mask_np)
        img_or = Image.fromarray(img_np)

        img_slic = Image.fromarray(s_slic)
        brd_slic = Image.fromarray(b_slic)

        img_slic.save(f"./slic_img/segmentado/{i}.jpg")
        brd_slic.save(f"./slic_img/borde/{i}.jpg")

        imagen.save(f"./images/resultados/{i}.jpg")
        img_bordes.save(f"./images/bordes/{i}.jpg")
        img_mask.save(f"./images/mask/{i}.jpg")
        img_or.save(f"./images/original/{i}.jpg")

    return

if __name__ == '__main__':
    main()
