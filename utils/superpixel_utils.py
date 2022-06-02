import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, minmax_scale, normalize
from matplotlib import cm
import matplotlib as mpl
from sklearn.neighbors import kneighbors_graph
import networkx as nx


def HSI_to_superpixels(img, num_superpixel, is_pca=True, is_show_superpixel=False):
    n_row, n_col, n_band = img.shape
    if is_pca:
        pca = PCA(n_components=0.95)
        img = pca.fit_transform(scale(img.reshape(-1, n_band))).reshape(n_row, n_col, -1)
    superpixel_label = slic(img, n_segments=num_superpixel, compactness=20, max_iter=10, convert2lab=False,
                            enforce_connectivity=True, min_size_factor=0.3, max_size_factor=2, slic_zero=False)
    if is_show_superpixel:
        x = minmax_scale(img[:, :, :3].reshape(-1, 3)).reshape(n_row, n_col, -1)
        # color = (162/255, 169/255, 175/25)
        color = (132/255, 133/255, 135/255)
        # mask = mark_boundaries(x, superpixel_label, color=color, mode='subpixel')
        mask_boundary = find_boundaries(superpixel_label, mode='subpixel')
        mask_ = np.ones((mask_boundary.shape[0], mask_boundary.shape[1], 3))
        mask_[mask_boundary] = color
        plt.figure()
        plt.imshow(mask_)
        plt.axis('off')
        plt.show()
    return superpixel_label


def show_superpixel(label, x=None):
    color = (132 / 255, 133 / 255, 135 / 255)
    if x is not None:
        color = (162/255, 169/255, 175/25)
        x = minmax_scale(x.reshape(label.shape[0] * label.shape[1], -1))
        x = x.reshape(label.shape[0], label.shape[1], -1)
        mask = mark_boundaries(x[:, :, :3], label, color=(1, 1, 0), mode='outer')
    else:
        mask_boundary = find_boundaries(label, mode='subpixel')
        mask = np.ones((mask_boundary.shape[0], mask_boundary.shape[1], 3))
        mask[mask_boundary] = color
    fig = plt.figure()
    plt.imshow(mask)
    plt.axis('off')
    plt.tight_layout()
    fig.savefig('superpixel.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()


def create_association_mat(superpixel_labels):
    labels = np.unique(superpixel_labels)
    # print(labels)
    n_labels = labels.shape[0]
    print('num superpixel: ', n_labels)
    n_pixels = superpixel_labels.shape[0] * superpixel_labels.shape[1]
    association_mat = np.zeros((n_pixels, n_labels))
    superpixel_labels_ = superpixel_labels.reshape(-1)
    for i, label in enumerate(labels):
        association_mat[np.where(label == superpixel_labels_), i] = 1
    return association_mat


def create_spixel_graph(source_img, superpixel_labels):
    s = source_img.reshape((-1, source_img.shape[-1]))
    a = create_association_mat(superpixel_labels)
    # t = superpixel_labels.reshape(-1)
    mean_fea = np.matmul(a.T, s)
    regions = regionprops(superpixel_labels + 1)
    n_labels = np.unique(superpixel_labels).shape[0]
    center_indx = np.zeros((n_labels, 2))
    for i, props in enumerate(regions):
        center_indx[i, :] = props.centroid  # centroid coordinates
    ss_fea = np.concatenate((mean_fea, center_indx), axis=1)
    ss_fea = minmax_scale(ss_fea)
    adj = kneighbors_graph(ss_fea, n_neighbors=50, mode='distance', include_self=False).toarray()

    # # # show initial graph
    # import matplotlib.pyplot as plt
    # adj_ = np.copy(adj)
    # adj_[np.where(adj != 0)] = 1
    # plt.imshow(adj_, cmap='hot')
    # plt.show()

    # # auto calculate gamma in Gaussian kernel
    X_var = ss_fea.var()
    gamma = 1.0 / (ss_fea.shape[1] * X_var) if X_var != 0 else 1.0
    adj[np.where(adj != 0)] = np.exp(-np.power(adj[np.where(adj != 0)], 2) * gamma)

    # adj = euclidean_dist(ss_fea, ss_fea).numpy()
    # adj = np.exp(-np.power(adj, 2) * gamma)
    np.fill_diagonal(adj, 0)

    # show_graph(adj, center_indx)
    return adj, center_indx


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    import torch
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-8).sqrt()  # for numerical stability
    return dist

# def cosine_sim_with_temperature(x, temperature=0.5):
#     x = normalize(x)
#     sim = np.matmul(x, x.T) / temperature  # Dot similarity

def show_graph(adj, node_pos):
    plt.style.use('seaborn-white')
    D = np.diag(np.reshape(1./np.sum(adj, axis=1), -1))
    adj = np.dot(D, adj)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    G = nx.from_numpy_array(adj)
    for i in range(node_pos.shape[0]):
        G.nodes[i]['X'] = node_pos[i]
    # edge_weights = [(u, v) for u, v in G.edges()]
    pos = nx.get_node_attributes(G, 'X')
    # nx.draw(G, pos=pos, node_size=40, node_color='b', edge_color='black')  #  #fabebe # white
    nx.draw(G, pos=pos, node_size=40, node_color='#CD3700')  # #fabebe # white
    norm_v = mpl.colors.Normalize(vmin=0, vmax=adj.max())
    cmap = cm.get_cmap(name='PuBu')
    m = cm.ScalarMappable(norm=norm_v, cmap=cmap)
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=d['weight'] * 5, alpha=0.5, edge_color=m.to_rgba(d['weight']))
    # draw graph
    # nx.draw(G, pos=pos, node_size=40)
    # nx.draw(G, pos=pos, node_size=node_size, node_color=color)
    plt.show()

