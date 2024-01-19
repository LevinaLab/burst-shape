import os
#os.environ['OPENBLAS_NUM_THREADS'] ='40'
import numpy as np
import sklearn
import scipy
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


def calculate_dist_matrix(data, metric):
    """ Calculate pairwise distances for each point in dataset with given metric

        Args:
            data (nd.array): Array containing data (n x m)
            metric (string, or callable): one of sklearns pairwise metrics : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances

        Returns:
            dist_matrix (nd.array): matrix of pairwise distances for each datapoint
            sorted_dist_matrix (nd.array): indices for row sorting distance matrix
    """
    dist_matrix = pairwise_distances(data, data, metric = metric) # calculate pairwise distances
    sorted_dist_matrix = np.argsort(dist_matrix, axis=1)

    return dist_matrix, sorted_dist_matrix


def get_global_scaling_parameter(data):
    svd = TruncatedSVD(100 ,random_state=42)
    projection = svd.fit_transform(data)

    w = svd.explained_variance_ratio_
    v = svd.explained_variance_

    for i in range(100):
        sigma_sqr = np.sum((w * v)[:i]) / np.sum(w[:i])
        if sigma_sqr > 0.95:
            break
    return sigma_sqr

def get_local_scaling_parameter(data,dist_matrix, sorted_dist_matrix, k = 7):
    local_sigmas = np.zeros(len(data))
    for i in range(len(data)):
        kth_nn = sorted_dist_matrix[i,k]
        local_sigmas[i] = dist_matrix[i,kth_nn]
    return local_sigmas



def get_local_scaled_affinity_matrix(data, metric="euclidean", k=7):
    dist_matrix, sorted_dist_matrix = calculate_dist_matrix(data, metric=metric)
    local_sigmas = get_local_scaling_parameter(data,dist_matrix, sorted_dist_matrix, k = k)

    A = np.zeros(dist_matrix.shape)

    for i in range(len(data)):
        A[i] = np.exp(-(dist_matrix[i])/(local_sigmas[i] * local_sigmas))
    np.fill_diagonal(A, 0)

    sorted_indices = np.argsort(A, axis=1)

    return A, sorted_indices





def construct_knn_graph(matrix,sorted_indices,matrix_info,k=10, mutual = False,weighting = True):
    """ Constuct KNN Graph from distance/similarity matrix

    Args:
        matrix (nd.array): nxn matrix containing pairwise distances/similarities
        sorted_indices (nd.array): nxn matrix of indices to sort rows of distance/dimilarity matrix in descending order
        k (int): number of neighbours to take into account
        mutual (bool): Wether to construct knn in mutual manner or not. Mutual in sense of: j beeing in k-nearest neighbours of i does not imply that i is in k-nearest neighbours of j. All vertices in a mutual k-NN graph have a degree upper-bounded by k.
        weighting (str): indicate wether matrix contains of similarities or distances

    Returns:
        A (nd.arrays): Adjacency matrix of the knn-graph
    """
    A = np.zeros(matrix.shape)

    if mutual:
        if matrix_info == "similarity":
            print("Build mutual KNN-Graph based on Similarity of data points!")
        else:
            print("Build mutual KNN-Graph based on Distance of data points!")
    else:
        if matrix_info == "similarity":
            print("Build symmetric KNN-Graph based on Similarity of data points!")
        else:
            print("Build symmetric KNN-Graph based on Distance of data points!")

    print("Weighting:", weighting)
    if mutual: # knn graph only when among both knn connect
        for i, indices in enumerate(sorted_indices):
            if matrix_info == "similarity":
                k_nearest = indices[-(k+1) : -1]
            else: #matrix_info == "distance":
                k_nearest = indices[1:k+1]
            for j in k_nearest:
                if i in sorted_indices[j,1:k+1]:
                    if weighting:
                        A[i,j] = matrix[i,j]
                    else:
                        A[i,j] = 1
    else:

        for i,indices in enumerate(sorted_indices):
            if matrix_info == "similarity":
                k_nearest = indices[-(k+1) : -1]
            else: # matrix_info == "distance":
                k_nearest = indices[1:k + 1]

            if weighting:
                A[i,k_nearest] = matrix[i,k_nearest]
                A[k_nearest,i] = matrix[k_nearest,i]
            else:
                A[i,k_nearest] = 1
                A[k_nearest, i] = 1
    return A


def calculate_laplacian(A, normalize = True,normalization_type = None,reg_lambda = 0.1,use_lambda_heuristic = False, saving_lambda_file = "data/quin_rohe_heuristic_lambda",   saving = False, saving_file = "data/L_norm"):
    """ Calculate the graph Laplacian for given KNN-Graph

    Args:
        A (nd.array): Adjacency Matrix of a knn-Graph
        normalize(bool): Wether to normalize Laplacian
        normalization_type (str): which approach to use for normalization
        reg_lambda (int): hyperparameter for regularization strength
        use_lambda_heuristic (bool): apply Qin Rohe heuristic for lambda
        saving_lambda_file:  File name for saving
        saving (bool): True if you want to save matrices for each folds
        saving_name (str): File name for saving
    Returns:
        L (nd.arrays): (normalized) graph Laplacian
    """


    print("Calculate Normalized Laplacians")

    # calcualte normalized Laplacian
    n = A.shape[0] # get number of data points in KNN-Graph
    if use_lambda_heuristic:
        test = np.sum(A,axis = 1)
        #print(test.shape)
        #print(test)
        reg_lambda = np.mean(np.sum(A,axis = 1)) # Tai Qin and Karl Rohe. 2013 heuristic
        print("Use Qin & Rohe heuristic for regularizaton!")
        np.save(saving_lambda_file, reg_lambda)

    if reg_lambda:
        print("Apply regularization!")
        print("lamda = %.4f" % reg_lambda)
        A = A + (reg_lambda/n * np.ones((n,n))) # apply regularization [Zhang and Rohe, 2018]

    D = np.sum(A,axis = 1) # get vertices degree

    D_inv = np.reciprocal(D)
    D_inv[np.where(np.isinf(D_inv))] = 0
    D_inv_sqrt = np.reciprocal(np.sqrt(D))
    D_inv_sqrt[np.where(np.isinf(D_inv_sqrt))] = 0 #division by zero
    D = np.diag(D)
    D_inv_sqrt = np.diag(D_inv_sqrt)
    D_inv = np.diag(D_inv)

    if normalize:
        if normalization_type == "symmetric":
            print("Normalization: " +  normalization_type)
            L = D_inv_sqrt @ (D-A) @ D_inv_sqrt # calculate normalized laplacian [Ng et al. 2002]
        if normalization_type == "random-walk": #
            print("Normalization: " +  normalization_type)
            L = D_inv @ (D-A)

    else:
        L = D - A


    if saving:
        np.save(saving_file,L)

    return L

def calculate_eigenvectors_and_values(L, saving = False, saving_file= "data/"):
    """ Calculate the eigenvectors and eigenvalues graph Laplacian

    Args:
        L (nd.arrays): (normalized) graph Laplacian
        saving (bool): True if you want to save matrices for each folds
        saving_name_eigenvectors (str): File name for saving
        saving_name_eigenvalues (str): File name for saving

    Returns:
        eigvec (nd.arrays): eigenvectors of graph Laplacian
        eigval (nd.arrays): eigenvalues of graph Laplacian
    """

    print("Calculate Eigenvalues and Vectors of Laplacian")

    # calcualte eigenvalues and eigenvector
    eigval, eigvec  = scipy.linalg.eigh(L)
    idx = eigval.argsort()#[::-1] # sort eigenvalues and corresponding eigenvectors in ascending order

    eigval = eigval[idx]
    eigvec = eigvec[:,idx]


    if saving:
        np.save(saving_file + "eigenvalues", eigval)
        np.save(saving_file + "eigenvectors", eigvec)


    return eigvec, eigval


def cluster_eigenvector_embedding(eigenvec, n_cluster):
    """ Cluster eigenvector embedding

    Args:
        eigenvec (nd.arrays): Eigenvectors of Graph Laplacian
        n_cluster (int): number of clusters

    Returns:
        labels (list): list of cluster labels for each point
    """

    U = eigenvec[:,:n_cluster] # take first n_cluster eigenvectors into account building a matrix of n X n_clusters
    U = U.astype("float")
    T = sklearn.preprocessing.normalize(U, norm='l2') # row normalize matrix

    X = T
    kmeans = KMeans(n_clusters=n_cluster).fit(X) # apply k-means to cluster eigenvector embedding
    labels = kmeans.labels_
    return labels


def spectral_clustering(data, metric, metric_info,n_clusters, precomputed_matrix=None, k=5, mutual = False, weighting = False, normalize = True, normalization_type = "symmetric", use_lambda_heuristic = False, reg_lambda = 0.1, saving_lambda_file= "data/quin_rohe_heuristic_lambda",save_laplacian = False, save_eigenvalues_and_vectors = False):
    """ Cluster data into n_clusters using spectral clustering  based on eigenvectors of knn-graph laplacian

    Args:
        data (nd.array): Array containing data (n x m)
        metric (string, or callable): one of sklearns pairwise metrics : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances
        n_cluster (list): list of number of n_clusters
        metric_info (string): indicate whether matrix contains similarities or distances
        k (int): number of neighbours to take into account
        mutual (bool): Whether to construct knn in mutual manner or not. Mutual in sense of: j beeing in k-nearest neighbours of i does not imply that i is in k-nearest neighbours of j. All vertices in a mutual k-NN graph have a degree upper-bounded by k.
        weighting (bool): weighted edges in KNN - Graph
        normalize(bool): Whether to normalize Laplacian
        normalization_type (str): which approach to use for normalization
        use_lambda_heuristic (bool): apply Qin Rohe heuristic for lambda
        saving_lambda_file:  File name for saving
        reg_lambda (int): hyper-parameter for regularization strength

        save_laplacian (bool): True if you want to save Laplacian
        save_eigenvalues_and_vectors (bool): True if you want to save eigenvectors and eigenvalues
        save_labels (bool): True if you want to save labels

    Returns:
        labels_per_n_clusters (lists of list): list of lists containing the cluster labels for each point in data set
    """


    if metric == "local_scaled_affinity":
        print("Calculate local scaled affinity matrix for constructing KNN-Graph")
        dist_matrix, sorted_dist_matrix = get_local_scaled_affinity_matrix(data,k=7)

    elif metric == "precomputed":
        print("Use precomputed matrix for constructing KNN-Graph")
        dist_matrix = precomputed_matrix
        sorted_dist_matrix = np.argsort(precomputed_matrix,axis=1)
    else:
        print("Calculate %s matrix for constructing KNN-Graph" % metric)
        dist_matrix, sorted_dist_matrix = calculate_dist_matrix(data, metric)

    A = construct_knn_graph(dist_matrix,sorted_dist_matrix,metric_info, k=k, mutual = mutual, weighting = weighting)

    L = calculate_laplacian(A, normalize = normalize,normalization_type = normalization_type, reg_lambda=reg_lambda, use_lambda_heuristic=use_lambda_heuristic, saving_lambda_file=saving_lambda_file, saving = save_laplacian, saving_file = "data/L_norm")

    eigvec, eigval = calculate_eigenvectors_and_values(L, saving = save_eigenvalues_and_vectors, saving_file= "data/")

    labels_per_n_clusters = []
    for n_cluster in n_clusters:
        labels = cluster_eigenvector_embedding(eigvec, n_cluster)
        labels_per_n_clusters.append(labels)

    return labels_per_n_clusters, eigvec, eigval