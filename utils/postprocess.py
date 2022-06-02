import numpy as np
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import networkx as nx


def spixel_to_pixel_labels(sp_level_label, association_mat):
    sp_level_label = np.reshape(sp_level_label, (-1, 1))
    pixel_level_label = np.matmul(association_mat, sp_level_label).reshape(-1)
    return pixel_level_label.astype('int')


def affinity_to_pixellabels(affinity_mat, n_clusters):
    # Coef = thrC(affinity_mat, 0.8)
    # y_pre, C = post_proC(Coef, n_clusters, 8, 18)

    affinity_mat = 0.5 * (np.abs(affinity_mat) + np.abs(affinity_mat.T))
    # import matplotlib.pyplot as plt
    # print(affinity_mat.diagonal().sum())
    # plt.imshow(affinity_mat, cmap='hot')
    # plt.show()

    # # using pure SC
    spectral = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize', random_state=42)
    spectral.fit(affinity_mat)
    y_pre = spectral.fit_predict(affinity_mat)
    # kmeans = cluster.KMeans(n_clusters=n_clusters, max_iter=500, random_state=42)
    # y_pre = kmeans.fit_predict(affinity_mat) + 1

    y_pre = y_pre.astype('int')
    return y_pre


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp

def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs, 0)
    for i in range(N):
        Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
    Cksym = Cabs + Cabs.T
    return Cksym


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L