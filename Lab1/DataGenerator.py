import numpy as np

def norm_dataset(mu, sigma, N):
    mu0 = mu[0]
    mu1 = mu[1]
    sigma0 = sigma[0]
    sigma1 = sigma[1]
    col = len(mu0)

    class0 = np.random.normal(mu0[0], sigma0[0], [N, 1])
    class1 = np.random.normal(mu1[0], sigma1[0], [N, 1])

    for i in range(1, col):
        v0 = np.random.normal(mu0[i], sigma0[i], [N, 1])
        class0 = np.hstack((class0, v0))

        v1 = np.random.normal(mu1[i], sigma1[i], [N, 1])
        class1 = np.hstack((class1, v1))

    Y1 = np.ones((N, 1), dtype=bool)
    Y0 = np.zeros((N, 1), dtype=bool)

    X = np.vstack((class0, class1))
    Y = np.vstack((Y0, Y1)).ravel()

    rng = np.random.default_rng()
    arr = np.arange(2 * N)
    rng.shuffle(arr)

    X = X[arr]
    Y = Y[arr]

    return X, Y, class0, class1


#решение самостоятельного задания вариант 10
def nonlinear_dataset_10(N, seed=None):

    rng = np.random.default_rng(seed)
    theta_min, theta_max = np.deg2rad(180), np.deg2rad(270) #угол в радианах где будут все точки
  
    #функция для рисования дуг
    def rounded_arc(cx, cy, r_min, r_max, n):
        r_mid, r_half = (r_min + r_max) / 2, (r_max - r_min) / 2
        #разделение точек на три части (n_cap на закругленные части, а n_body на основную часть)
        n_cap = int(n * 0.12)
        n_body = n - 2 * n_cap
 
        #рисуем тело дуги
        theta = rng.uniform(theta_min, theta_max, n_body) 
        r = rng.uniform(r_min, r_max, n_body)
        x = cx - r * np.cos(theta)
        y = cy + r * np.sin(theta)
 
        #рисуем точки внутри обычного пятна
        for edge in (theta_min, theta_max):
            ex, ey = cx - r_mid * np.cos(edge), cy + r_mid * np.sin(edge)
            ang = rng.uniform(0, 2 * np.pi, n_cap)
            rad = r_half * np.sqrt(rng.uniform(0, 1, n_cap))
            x = np.concatenate([x, ex + rad * np.cos(ang)])
            y = np.concatenate([y, ey + rad * np.sin(ang)])
 
        return np.column_stack((x, y))
 
    class0 = rounded_arc(cx=-1, cy=1, r_min=2.0, r_max=2.8, n=N)
    class1 = rounded_arc(cx=0, cy=0, r_min=2.0, r_max=2.8, n=N)
 
    X = np.vstack((class0, class1))
    Y = np.concatenate([np.zeros(N, dtype=bool), np.ones(N, dtype=bool)])
 
    order = rng.permutation(2 * N)
    return X[order], Y[order], class0, class1