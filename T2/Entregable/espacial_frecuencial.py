import numpy as np
import fouriere as fr

def DoG(sigma1, sigma2, A, B, k=3):
    sigma_e1 = 1.0 / (2.0 * np.pi * sigma1)
    sigma_e2 = 1.0 / (2.0 * np.pi * sigma2)

    sigma_max_espacial = max(sigma_e1, sigma_e2)
    radio = int(k * sigma_max_espacial)
    if radio < 1: radio = 1

    ax = np.arange(-radio, radio + 1)
    xx, yy = np.meshgrid(ax, ax)
    r_cuadrado = xx ** 2 + yy ** 2

    term1 = (np.sqrt(2 * np.pi) * sigma1 * A) * np.exp(-2 * (np.pi ** 2) * (sigma1 ** 2) * r_cuadrado)
    term2 = (np.sqrt(2 * np.pi) * sigma2 * B) * np.exp(-2 * (np.pi ** 2) * (sigma2 ** 2) * r_cuadrado)

    kernel = term1 - term2
    return kernel

def gauss2D(sigma, k = 3):
    radio = int(k * sigma)
    ax = np.arange(-radio, radio + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

def laplaciano():
    return np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0]
    ])

def frecuencia(kernel, P, Q):
    km, kn = kernel.shape
    hp = np.zeros((P, Q), dtype=float)

    hp[:km, :kn] = kernel

    hp = np.roll(hp, shift=(-(km // 2), -(kn // 2)), axis=(0, 1))

    x, y = np.meshgrid(np.arange(Q), np.arange(P))
    hp = hp * ((-1) ** (x + y))

    #Hp = fr.fourier_transform(hp)
    return np.fft.fft2(hp)

