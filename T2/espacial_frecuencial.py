import numpy as np
import fouriere as fr

def DoG(sigma1, sigma2, k=3):
    sigma_pequeno = min(sigma1, sigma2)
    sigma_grande = max(sigma1, sigma2)

    radio = int(k * sigma_grande)
    ax = np.arange(-radio, radio + 1)
    xx, yy = np.meshgrid(ax, ax)

    kernel1 = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_pequeno ** 2))
    kernel1 = kernel1 / kernel1.sum()

    kernel2 = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_grande ** 2))
    kernel2 = kernel2 / kernel2.sum()

    return kernel1 - kernel2

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

def frecuencia(kernel, M, N):
    P = M
    Q = N
    hp = np.zeros((P, Q), dtype=complex)

    km, kn = kernel.shape

    hp[:km, :kn] = kernel

    # 3. EL TRUCO MÁGICO: "Enrollamos" la matriz para que el centro
    # del kernel quede posicionado exactamente en la coordenada (0,0)
    hp = np.roll(hp, shift=(-(km // 2), -(kn // 2)), axis=(0, 1))

    # 4. Centramos el espectro resultante multiplicando por (-1)^(x+y)
    #x, y = np.meshgrid(np.arange(Q), np.arange(P))
    #hp *= (-1) ** (x + y)

    Hp = fr.fourier_transform(hp)
    return Hp

