import helper 

if __name__ == '__main__': 
    #Runs a single instance of WDl with same parameters as used in experiment
    #dir_name parameter is directory save name
    helper.wdl_instance(k=32, train_size=2000, dir_name='random_data', 
                        reg=0.1, mu=0.0001, max_iters=400, n_restarts=2, 
                        mode='true_random++')