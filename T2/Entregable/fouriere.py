import numpy as np

def fourier_transform(image, i = 1):
    m, n = image.shape

    F = np.zeros((m,n), dtype=complex)
    for q in range (m):
        v = image[q,:]
        F[q,:] = fft(v,i)
    for r in range (n):
        v = F[:,r]
        F[:,r] = fft(v,i)

    return F

def fft(vector, i = 1):
    n = len(vector)
    if n <= 1:
        return vector
    w_n = np.exp((i*2.0j) * np.pi / vector.size)
    w = 1
    vector_f0 = vector[0::2]
    vect0r_f1 = vector[1::2]
    f_0 = fft(vector_f0,i)
    f_1 = fft(vect0r_f1,i)
    f = [0] * n

    mitad = int(n/2)
    for k in range(mitad) :
        termino = w*f_1[k]
        f[k] = f_0[k] + termino
        f[k + mitad] = f_0[k] - termino
        w *= w_n
    return f