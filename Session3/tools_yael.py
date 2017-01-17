import sys
import time
import numpy as np
from yael import ynumpy
        
    
##############################################################################
def predict_fishergmm(gmm, des, options):
    # Compute the Fisher Vectors from the features.
    # des is supposed to be the features of a single image.
    des2 = np.float32(des)
    fisher = ynumpy.fisher(gmm, des2, include = ['mu','sigma'])
    return fisher
    

##############################################################################
def compute_codebook_gmm(kmeans, D):
    # Clustering (unsupervised classification)
    # Fit a GMM over the features.
    print 'Computing gmm with ' + str(kmeans) + ' centroids'
    sys.stdout.flush()
    init = time.time()
    gmm = ynumpy.gmm_learn(np.float32(D), kmeans)
    end = time.time()
    print 'Done in ' + str(end-init) + ' secs.'
    return gmm