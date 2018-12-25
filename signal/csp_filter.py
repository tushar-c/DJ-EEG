import numpy as np 


def organize_data(features, labels, n_classes=2):
    class_features = {i: [] for i in range(n_classes)}
    class_covariances = []

    N = len(features)
    nrows, ncols = features[0].shape

    for n in range(N):
        X = features[n]
        y = labels[n]
        class_features[y].append(X)

    for c in range(n_classes):
        cov = np.zeros((nrows, nrows))
        for mat in class_features[c]:
            cov += np.matmul(mat, mat.T)
        cov /= len(class_features[c])
        class_covariances.append(cov)

    return class_covariances


def general_eigval(pooled_covs):
    A, B = pooled_covs[0], pooled_covs[1]
    C = np.matmul(np.linalg.inv(B), A)
    eigs = np.linalg.eig(C)
    soln_eigval = eigs[0]
    soln_eigvec = eigs[1]
    D = soln_eigvec
    E = np.diag(soln_eigval)
    return D, E


def project_to_csp_space(filter_mat, data_mat):
    return np.matmul(filter_mat.T, data_mat)



class1 = [np.random.randn(100, 25) for j in range(20)]
class2 = [np.random.randn(100, 25) for k in range(20)]

labels1 = [0 for i in range(20)]
labels2 = [1 for i in range(20)]

class1.extend(class2)
labels1.extend(labels2)

features = [k for k in class1]
labels = [l for l in labels1]

covs = organize_data(features, labels)
eigvec_mat, eigval_mat = general_eigval(covs)
csp_space = project_to_csp_space(eigvec_mat, class1)

