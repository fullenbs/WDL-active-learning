import active
import numpy as np

if __name__ == '__main__': 
    #Carries out experiment
    n_clusters = [8, 10, 12, 14, 16, 18, 20, 30, 40, 50]
    n_points = [1, 2, 4, 5, 10, 15, 20]
    i = 0
    n = 1
    vals_true = np.zeros((len(n_clusters), n))
    vals_inpaint = np.zeros((len(n_clusters), n))

    for c in n_clusters: 
        for p in n_points: 
            k = 0
            for r in range(0, n): 
                #NOTE: CHANGE dir_name and save_name to make run 
                #dir_name=directory where atoms/weights are loaded from
                #save_name=where you want results saved to 
                #tsne: Set true if you want plots, false if you don't want plots
                #Outputs are k: which is true accuracy and l which is point relabel/inpaint
                (k, l) = active.wadal_control(num_clusters=c, n_points=p, tsne=True, 
                                         dir_name='Random_data_1', save_name='whisper_testing')
                vals_true[i, r] = k
                vals_inpaint[i, r] = l
        i += 1
    print(np.mean(vals_true, axis=1), np.std(vals_true, axis=1))
    print(np.mean(vals_inpaint, axis=1), np.std(vals_inpaint, axis=1))