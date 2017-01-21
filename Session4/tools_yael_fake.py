import sys

# Fake module, for allowing running the system in Windows.
# This should actually never be called using Windows. It will give
# an error if called.
        
    
##############################################################################
def predict_fishergmm(gmm, des, options):
    print 'Error: Using Fisher Vectors is not possible in Windows.'
    sys.stdout.flush()
    sys.exit()
    
    

##############################################################################
def compute_codebook_gmm(kmeans, D):
    print 'Error: Using Fisher Vectors is not possible in Windows.'
    sys.stdout.flush()
    sys.exit()