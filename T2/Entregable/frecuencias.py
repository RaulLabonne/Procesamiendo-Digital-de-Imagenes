import numpy as np
import fouriere

class frecuencias:

    def __init__(self, img, max_kernel_size=150):
        M,N = img.shape
        self.__M = M
        self.__N = N

        min_P = M + max_kernel_size - 1
        min_Q = N + max_kernel_size - 1

        self.P = 2 ** int(np.ceil(np.log2(min_P)))
        self.Q = 2 ** int(np.ceil(np.log2(min_Q)))

        self.__img = np.zeros((self.P,self.Q), dtype=complex)
        self.__img[:M, :N] = img

        xv, yv = np.meshgrid(np.arange(self.Q), np.arange(self.P))
        self.__centrado = (-1.0) ** (xv + yv)
        self.__img = self.__img * self.__centrado

        ## dft
        #self.__F = fouriere.fourier_transform(self.__img) 
        self.__F = np.fft.fft2(self.__img)
        u = np.arange(self.P) - self.P // 2
        v = np.arange(self.Q) - self.Q // 2
        V,U = np.meshgrid(v, u)
        self.__D = np.sqrt(U ** 2 + V**2)

    def pasa_banda(self, H):
        ## Filtro y producto
        G = H * self.__F

        espectro = np.log(np.abs(G) + 1)

        gp = np.real(np.fft.ifft2(G)) * self.__centrado
        return gp[:self.__M, :self.__N], espectro


    def pasa_banda_ideal(self, w, fc):
        return ((self.__D >= fc - w / 2) & (self.__D <= fc + w / 2)).astype(float)

    def pasa_banda_Butter(self, w, n, fc ):
        D_safe = np.where(self.__D == 0, 1e-6, self.__D)
        ratio = (D_safe ** 2 - fc ** 2) / (D_safe * w + 1e-6)
        return 1 / (1 + ratio ** (2 * n))

    def pasa_banda_Gauss(self, w, fc):
        D_safe = np.where(self.__D == 0, 1e-6, self.__D)
        return np.exp(-((D_safe ** 2 - fc ** 2) ** 2) / (D_safe ** 2 * w ** 2 + 1e-6))

    def H_dog(self, sigma1, sigma2,A,B):
        return A * np.exp(-self.__D ** 2 / (2 * sigma1 ** 2)) - B * np.exp(-self.__D ** 2 / (2 * sigma2 ** 2))

