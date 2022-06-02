import numpy as np
from sklearn import metrics
from munkres import Munkres
# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def evaluate(label, pred, is_HSI_score=False):
    # if not is_labeled_pixel:
    #     indx_nonzero = np.where(label != 0)
    #     pred = pred[indx_nonzero]
    #     label = label[indx_nonzero]
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    if is_HSI_score:
        pur = purity_score(label, pred_adjusted)
        ca = class_acc(label, pred_adjusted)
        kappa = metrics.cohen_kappa_score(label, pred_adjusted)
        return acc, kappa, nmi, ari, pur, ca
    return nmi, ari, f, acc


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def class_acc(y_true, y_pre):
    """
    calculate each class's acc
    :param y_true:
    :param y_pre:
    :return:
    """
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
        y_c_p = y_pre[np.nonzero(y_true == c)]
        acurracy = metrics.accuracy_score(y_c, y_c_p)
        ca.append(acurracy)
    ca = np.array(ca)
    return ca


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# # Alternative version
def cluster_accuracy(y_true, y_pre, return_aligned=False):
    y_true = y_true.astype('float32')
    y_pre = y_pre.astype('float32')
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pre == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]

    # # calculate accuracy
    err_x = np.sum(y_true[:] != y_best[:])
    missrate = err_x.astype(float) / (y_true.shape[0])
    acc = 1. - missrate
    nmi = metrics.normalized_mutual_info_score(y_true, y_pre)
    kappa = metrics.cohen_kappa_score(y_true, y_best)
    ca = class_acc(y_true, y_best)
    ari = metrics.adjusted_rand_score(y_true, y_best)
    fscore = metrics.f1_score(y_true, y_best, average='micro')
    pur = purity_score(y_true, y_best)
    if return_aligned:
        return y_best, acc, kappa, nmi, ari, pur, ca
    return acc, kappa, nmi, ari, pur, ca


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('=====Model Parameters====')
    print('Total:{:.3f}M,\nTrainable: {:.3f}M'.format(total_num/1e6, trainable_num/1e6))