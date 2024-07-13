import numpy as np 
import helper
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.manifold import TSNE
from statistics import mode


#Modified mask helper function
#Mask where any point in idx is 1, 0 otherwise
def labeled_mask(idx): 
    data = np.zeros(83*86)
    for i in range(0, idx.shape[0]): 
        data[int(idx[i])] = 1
    data = np.reshape(data, (83, 86))
    return data

#Core algorithm 
#embed: TSNE embedding
#n_clusters: # of clusters
#n_points: # of points per cluster
#idx: Indicies of points in data
#gt_data: Ground truth 
def wadal(embed, n_clusters, n_points, idx, gt_data): 
    #Does clustering and gets centroids
    km = KMeans(init='k-means++', n_init='auto', n_clusters=n_clusters)
    km.fit(embed)
    labels = km.labels_
    centroid = km.cluster_centers_

    #Gets the distance from centroids to each point in embedding
    vals = distance.cdist(centroid, embed)
    label_remap = np.zeros(n_clusters) #Stores clusters

    #Finds the N closest points to each centroid
    for row in range(vals.shape[0]):

        #Technically, if N or K are too large, it's possible that we include 
        #points outside the cluster. However, as the goal is for this to work 
        #at low values of N and K, this is a moot point. 
        labeled = 0
        dist_track = vals[row,:]
        args = np.argsort(dist_track)
        j = 0         
        while labeled < n_points: 
            if gt_data[idx[args[j]]] != 0: 
                labeled += 1
            j += 1
        
        #This logic builds a map for each kmeans cluster to the most popular label among 
        #those N points.
        args_label = gt_data[idx[args[0:j]]] 
        args_label = args_label[args_label != 0] 
        gt_label = np.zeros(n_points)
        for k in range(0, args_label.shape[0]): 
            gt_label[k] = args_label[k]
        label_remap[row] = mode(gt_label) 

    #Carries out the map and returns the learned labels
    for i in range(len(labels)): 
        labels[i] = label_remap[labels[i]]
    return labels

#Control loop for WADAL, algorithm carried out in wadal()
def wadal_control(num_clusters=20, n_points=5, tsne=True, dir_name='Random_data_1',
                  save_name='whisper_test'):
    #Generic mapping for coloring
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}

    (gt_data, mask) = helper.gt_and_mask(remap)

    X = torch.load(dir_name + '/coeff.pt').numpy() #Learned coefficients
    idx = torch.load(dir_name + '/train_index.pt').numpy()

    #Calculates and appends mass to learned coefficients
    data_size = idx.shape[0]
    data = helper.data_loader('data')
    mass = np.zeros(data_size)
    for i in range(data_size): 
        mass[i] = np.sum(data[idx[i],:])
    mass = np.reshape(mass, (1, data_size))
    X = np.append(X.T, mass.reshape(-1, 1), axis=1)
    #X = X.T
    for i in range(X.shape[0]):
        X[i,:] /= np.sum(X[i,:])

    #Sets proper colormap
    cmap = cm.get_cmap('viridis', 7)
    new_cmap = mcolors.ListedColormap(cmap.colors)
    new_cmap.colors[0] = (1, 0, 0, 1)

    embed = TSNE(n_components=2, learning_rate='auto', init='random', 
            perplexity=25).fit_transform(X)
    if tsne:
        plt.scatter(embed[:,0], embed[:,1], c=gt_data[idx], cmap=new_cmap)
        plt.savefig(save_name + '/2_embedding_n_clusters=' + str(num_clusters) + '_n_points=' + 
                    str(n_points) + '.pdf', bbox_inches='tight')
        plt.clf()    

    #Active learning step
    labels = wadal(embed, num_clusters, n_points, idx, gt_data)

    #Accuracy of labeling, exclude unlabeled data from these results
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
        plt.savefig(save_name + '/2_post_algo_n_clusters=' + str(num_clusters) + '_n_points='
                 + str(n_points) + '_acc=' + str(round(acc, 2)) + '.pdf', bbox_inches='tight')
        plt.clf()

    #Spatial relabeling + inpainting process
    train_plot = np.reshape(train_plot, (83, 86))
    relabel = helper.spatial_NN(train_plot, 10, mask, run_mode='relabel')
    relabel2 = helper.spatial_NN(relabel, 10, mask, run_mode='NN')

    #Determines accuracy
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
    return (acc, paint_acc)


#Carries out 1d algorithm but just on mass
def mass_cluster(): 
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    dir_name = 'Random_data_1'

    (gt_data, mask) = helper.gt_and_mask(remap)

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
    labels = wadal(embed, 8, 1, idx, gt_data)
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
