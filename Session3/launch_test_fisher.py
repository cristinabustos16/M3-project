# It is intended to be called like an array job, with the following command:
# qsub -cwd -V -S /home/1242538/virtualenvs/mlcv/bin/python -q fast.master.q@compute-0-2 -l mem_token=1000M,mem_free=1000M launch_test_fisher.py
import os

if __name__ == '__main__':

    # Fixed parameters:
    param1 = 'test_fv_'
    param2 = 1            # compute_codebook
    param3 = 32           # kmeans
    param4 = 1            # compute_subsets
    param5 = 100          # nfeatures SIFT
    param6 = 1            # apply dense sampling
    param7 = 8            # dense sampling step=radius
    param8 = 1            # apply spatial pyramids
    param9 = 2            # spatial pyramids depth
    param10 = '3x1'       # spatial pyramids configuration

    cmd_mask = "python session3_fisher_cluster.py %s %s %s %s %s %s %s %s %s %s"
    report_filename = param1 + str(param8) + '.txt'
    cmd = cmd_mask % (report_filename, str(param2), str(param3), str(param4), str(param5), str(param6), str(param7), str(param8), str(param9), param10)
    print cmd
    print os.popen(cmd).read()

#############################################################################