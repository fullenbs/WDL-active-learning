import numpy as np 
import helper
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.spatial import distance
from sklearn.manifold import TSNE
from statistics import mode

def kmeans_init(data, k, idx_track):
    d = data.shape[0]
    n = data.shape[1]
    C = np.zeros((d, k))

    idx_track = list(idx_track)
    idxes = list(range(n))
    idx_temp = np.zeros(k)
    a_1 = np.random.choice(n, 1)[0]
    idx_temp[0] = idx_track[int(a_1)]
    C[:,0] = data[:,a_1]

    for i in range(1, k):
        p = np.zeros(n - i)

        # compute distances to centroids
        for j in range(n - i):
            idx = idxes[j]
            p[j] = 100000
            for l in range(0, k): 
                d = np.linalg.norm(C[:,l] - data[:,idx])
                if d < p[j]:
                    p[j] = d
        p /= p.sum()

        # pick new centroid
        new_centroid_idx = np.random.choice(n - i, 1, p=p)[0]
        idx_temp[i] = idx_track[int(new_centroid_idx)]
        C[:,i] = data[:, new_centroid_idx]
        del idxes[new_centroid_idx]
        del idx_track[new_centroid_idx]
    return (C, idx_temp)

#Mask that is 1 if data is in set, 0 otherwise
def labeled_mask(idx): 
    data = np.zeros(83*86)
    for i in range(0, idx.shape[0]): 
        data[int(idx[i])] = 1
    data = np.reshape(data, (83, 86))
    return data

#TODO: Build up spectral clustering on embedding
def over_cluster(embed, n_clusters, n_points, idx, gt_data): 
    km = KMeans(init='k-means++', n_init='auto', n_clusters=n_clusters)
    #Core assignment algorithm
    km.fit(embed)
    labels = km.labels_
    centroid = km.cluster_centers_
    vals = distance.cdist(centroid, embed)
    label_remap = np.zeros(n_clusters)
    for row in range(vals.shape[0]):
        labeled = 0
        dist_track = vals[row,:]
        args = np.argsort(dist_track)
        j = 0         #NOTE: Want first n labeled points? 
        while labeled < n_points: 
            if gt_data[idx[args[j]]] != 0: 
                labeled += 1
            j += 1
        args_label = gt_data[idx[args[0:j]]] 
        args_label = args_label[args_label != 0] 
        gt_label = np.zeros(n_points)
        for k in range(0, args_label.shape[0]): 
            gt_label[k] = args_label[k]
        label_remap[row] = mode(gt_label) 
    for i in range(len(labels)): 
        labels[i] = label_remap[labels[i]]
    return labels


def modified_mass_cluster(num_clusters=20, n_points=5, tsne=True):
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    dir_name = 'Random_data_1'

    (gt_data, mask) = helper.gt_and_mask(remap)

    X = torch.load(dir_name + '/coeff.pt').numpy()
    idx = torch.load(dir_name + '/train_index.pt').numpy()
    save_name = 'test'

    data_size = idx.shape[0]
    data = helper.data_loader('data')
    mass = np.zeros(data_size)

    for i in range(data_size): 
        mass[i] = np.sum(data[idx[i],:])
    mass = np.reshape(mass, (1, data_size))
    X = np.append(X.T, mass.reshape(-1, 1), axis=1)
    X = X.T

    cmap = cm.get_cmap('viridis', 7)
    new_cmap = mcolors.ListedColormap(cmap.colors)
    new_cmap.colors[0] = (1, 0, 0, 1)

    for i in range(X.shape[0]):
        X[i,:] /= np.sum(X[i,:])
    if tsne: 
        embed = TSNE(n_components=2, learning_rate='auto', init='random', 
                    perplexity=25).fit_transform(X)
        plt.scatter(embed[:,0], embed[:,1], c=gt_data[idx], cmap=new_cmap)
        #plt.title('Embedding of data (Red is unlabeled)')
        plt.savefig(save_name + '/2_embedding_n_clusters=' + str(num_clusters) + '_n_points=' + 
                    str(n_points) + '.pdf', bbox_inches='tight')
        plt.clf()    
    else: 
        embed = X

    labels = over_cluster(embed, num_clusters, n_points, idx, gt_data)

    #Accuracy of labeling, must exclude unlabeled data from these results
    acc = 0
    zero_count = 0
    train_plot = np.zeros(83*86)
    for i in range(data_size):
        t = idx[i]
        j = labels[i]
        train_plot[t] = j
        if gt_data[t] == 0: 
            zero_count += 1
        elif gt_data[t] == j:
            acc += 1
    acc = acc/(data_size - zero_count)

    print('Clusters: ', num_clusters, 'Points: ', n_points, 'Accuracy: ', acc)
    if tsne: 
        plt.scatter(embed[:,0], embed[:,1], c=labels, cmap='viridis')
        #plt.title('Post algorithm labels accuracy=' + str(acc))
        plt.savefig(save_name + '/2_post_algo_n_clusters=' + str(num_clusters) + '_n_points='
                 + str(n_points) + '_acc=' + str(round(acc, 2)) + '.pdf', bbox_inches='tight')
        plt.clf()

    train_plot = np.reshape(train_plot, (83, 86))
    relabel = helper.spatial_NN(train_plot, 10, mask, run_mode='relabel')
    relabel2 = helper.spatial_NN(relabel, 10, mask, run_mode='NN')

    X = relabel2.reshape(-1)
    paint_acc = 0
    count = 0
    for i in range(83*86):
        if X[i] != 0 and gt_data[i] != 0:
            count += 1
            if X[i] == gt_data[i]: 
                paint_acc +=1 
    paint_acc = paint_acc/count
    paint_display = str(round(paint_acc, 3))
    print('Post inpainting and relabeling accuracy: ', paint_display)
    plt.imshow(relabel2, cmap=helper.virid_modify())
    #plt.title('Post inpainting/relabeling=' + paint_display)
    plt.savefig(save_name + '/2_post_inpaint_n_clusters=' + str(num_clusters) + '_n_points='
                + str(n_points) + '_acc=' + paint_display + '.pdf', bbox_inches='tight')
    plt.clf()
    return paint_acc


#Carries out out our algorithm just on mass
def mass_cluster(): 
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    dir_name = 'Random_data_1'

    (gt_data, mask) = helper.gt_and_mask(remap)

    X = torch.load(dir_name + '/coeff.pt').numpy()
    idx = torch.load(dir_name + '/train_index.pt').numpy()
    save_name = 'testing'

    data_size = idx.shape[0]
    data = helper.data_loader('data')
    mass = np.zeros(data_size)

    for i in range(data_size): 
        mass[i] = np.sum(data[idx[i],:])
    mass = np.reshape(mass, (1, data_size))
    embed = TSNE(n_components=2, learning_rate='auto', init='random', 
        perplexity=25).fit_transform(mass.T)
    labels = over_cluster(embed, 8, 1, idx, gt_data)
    acc = 0
    zero_count = 0
    train_plot = np.zeros(83*86)
    for i in range(data_size):
        t = idx[i]
        j = labels[i]
        train_plot[t] = j
        if gt_data[t] == 0: 
            zero_count += 1
        elif gt_data[t] == j:
            acc += 1
    acc = acc/(data_size - zero_count)

    print('Clusters: ', 8, 'Points: ', 1, 'Accuracy: ', acc)
    plt.scatter(embed[:,0], embed[:,1], c=labels, cmap='viridis')
    plt.title('Post algorithm labels accuracy=' + str(acc))
    plt.savefig(save_name + '/mass_test_n_clusters=8_n_points=1_acc=' 
                + str(round(acc, 2)) + '.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':  
    n_clusters = [8]
    n_points = [1]
    i = 0
    n = 1
    vals = np.zeros((len(n_clusters), n))

    for c in n_clusters: 
        for p in n_points: 
            k = 0
            for r in range(0, n): 
                k = modified_mass_cluster(num_clusters=c, n_points=p, tsne=True)
                vals[i, r] = k
        i += 1
    print(np.mean(vals, axis=1), np.std(vals, axis=1))

