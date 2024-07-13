This code is related to research for a conference paper at WHISPERS on active learning of HSIs (publication will be linked). If you have any questions about the code, please contact me Scott Fullenbaum, at scottad638@gmail.com 

All code for the underlying WDL algorithm is attributed here:  
https://github.com/MarshMue/GeoWDL-TAG-ML-ICML-2023

This repo contains further info about clustering/unmixing as well:
Code here is associated with papers on WDL for hyperspectral image unmixing and clustering.

Installation: 
Recommend a Conda environment to maintain. Code related to learned WDL results was run on the Tufts HPC cluster. All other experiments were carried out locally. THis is by no means an exhaustive list
of all packages, but here are some of the more important ones where version may matter: 
1. ```numpy 1.23.5```
2. ```pytorch 1.13.1```
3. ```POT 0.8.2 ```
4. ```scikit-learn 1.2.1```

The code was also run using ```Python 3.9.15```

## Random WDL experiment: 

Run ```python3 randomized.py```

This will run WDL on 2000 data points under certain parameters. It will output these results into a directory called '''random_data'''. The directory name can be easily changed within 
the file. Also note, DO NOT RUN repeat experiments under the same directory one after another. It will overwrite the old directory. Also, running this on another HPC or such system will require 
system-dependant code. 

## Synthetic Experiments:
To generate the synthetic results with the two Gaussians run: 

```python3 synthetic.py```

This will create a directory called '''synthetic_tests''' and carry out synthetic experiments using two Salinas A spectra. Results will vary as spectra selection is random. 

# Replication of Active Learning results: 
To replicate the active learning results run: 

'''python3 active_exp.py''' 

Note to do this, you need to have WDL learned atoms/weights. Open up the script, and change the parameters on the function call on line 22 according to the comments. One indicates where data is loaded
in from, the other indicates where you want results saved to. 


# Other notes: 
All my results are stored in '''whisper_test'''. I've moved the plots we include to '''whisper_pics''' for simplicity. One _minor_ thing to note is within '''whisper_test''' you'll notice
the inpainting plots look off. Initially when these were inpainted and included the learned labels for unlabeled data. Though interesting, this made the results a bit confusing to look at
and interpret, so they are left off for other plots. The code currently excludes them. 
