import helper
import torch

if __name__ == '__main__': 
    torch.set_default_dtype(torch.float64) 

    #Parameters: 
    #reg: entropy
    #lm: If True makes a linear mixture
    #dir_name: save name
    #mode: Sets type of experiment, keep on salinas.
    #sample size: number of synthetic elements you want  - 1

    #NOTE: If it doesn't run, just make a directory with name=dir_name
    helper.synthetic_experiments(reg=0.001, lm=False, dir_name='synthetic_tests', 
                                 mode='salinas', sample_size=50)