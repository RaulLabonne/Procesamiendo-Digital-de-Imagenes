import numpy as np
import fouriere

class frecuencias:

    def __init__(self, img):
        M,N = img.shape
        self.__M = M
        self.__N = N
        self.__P = M * 2
        self.__Q = N * 2

        self.__img = np.zeros((self.__P,self.__Q), dtype=complex)
        self.__img[:M, :N] = img

        ## dft
        self.__F = fouriere.fourier_transform(self.__img)
        u = np.arange(self.__P) - self.__P // 2
        v = np.arange(self.__Q) - self.__Q // 2
        V,U = np.meshgrid(v, u)
        self.__D = np.sqrt(U ** 2 + V**2)

    def pasa_banda(self, H):
        ## Filtro y producto
        G = H * self.__F

        espectro = np.log(np.abs(G) + 1)
        img_espectro = self.normalizar(espectro)

        m, n = G.shape
        x, y = np.meshgrid(np.arange(n), np.arange(m))
        G = G * ((-1) ** (x + y))

        gp = np.real(fouriere.fourier_transform(G, -1)) / (self.__P * self.__Q)

        return gp[:self.__M, :self.__N], img_espectro

    def normalizar(self, gp):
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


    def pasa_banda_ideal(self, w, fc):
        return ((self.__D >= fc - w / 2) & (self.__D <= fc + w / 2)).astype(float)

    def pasa_banda_Butter(self, w, n, fc ):
        D_safe = np.where(self.__D == 0, 1e-6, self.__D)
        ratio = (D_safe ** 2 - fc ** 2) / (D_safe * w + 1e-6)
        return 1 / (1 + ratio ** (2 * n))

    def pasa_banda_Gauss(self, w, fc):
        D_safe = np.where(self.__D == 0, 1e-6, self.__D)
        return np.exp(-((D_safe ** 2 - fc ** 2) ** 2) / (D_safe ** 2 * w ** 2 + 1e-6))


