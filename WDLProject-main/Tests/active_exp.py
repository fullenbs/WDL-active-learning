import active
import numpy as np
import os 

if __name__ == '__main__': 
    #Carries out experiment
    n_clusters = [8, 10, 12, 14, 16, 18, 20, 30, 40, 50]
    n_points = [1, 2, 4, 5, 10, 15, 20]
    i = 0
    n = 1
    vals = np.zeros((len(n_clusters), n))
    for c in n_clusters: 
        for p in n_points: 
            k = 0
            for r in range(0, n): 
                k = active.wadal_control(num_clusters=c, n_points=p, tsne=True, 
                                         dir_name='Random_data_1', save_name='whisper_testing')
                vals[i, r] = k
        i += 1
    print(np.mean(vals, axis=1), np.std(vals, axis=1))