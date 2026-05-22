import slic
import numpy as np
from PIL import Image

def main():
    #image = Image.open("./assets/lena_std.jpg")
    #image = Image.open('./assets/1_22_s.bmp')
    image = Image.open('./assets/3.png')
    array = np.array(image)
    #array = np.full_like(array1, 255)

    nsp = 50
    salida, bordes = slic.slic(array, nsp)

    img_bordes = slic.dibujar_limites(array, bordes, [255,255,255])

    imagen = Image.fromarray(salida)
    img = Image.fromarray(img_bordes)
    imagen.save(f"./assets/final2.jpg")
    img.save(f"./assets/bordes2.jpg")

    return

if __name__ == '__main__':
    main()
