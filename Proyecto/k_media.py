import numpy as np

def k_medias(input, k):
    N, D = input.shape
    indices = np.random.choice(N,k,replace=False)
    M = input[indices].astype(float)

    epsilon = 1e-8
    iteracion = 0
    while iteracion < 10:
        distancias_cuadradas = np.linalg.norm(input[:, np.newaxis] - M, axis=2) ** 2

        etiquetas = np.argmin(distancias_cuadradas, axis=1)

        M_anterior = np.copy(M)

        for i in range(k):
            puntos_cluster = input[etiquetas == i]
            if len(puntos_cluster) > 0:
                M[i] = np.mean(puntos_cluster, axis=0)
            else:
                M[i] = input[np.random.choice(N)]
        diferencias = M - M_anterior

        S = np.sum(np.linalg.norm(diferencias, axis=1))

        if S <= epsilon:
            break
        iteracion += 1
    return etiquetas, M

