# Program to try many different costs for SVM in parallel.
# It is intended to be called like an array job, with the following command:
# qsub -cwd -V -S /usr/bin/python -q fast.master.q@compute-0-2 -t 1-10 -l mem_token=500M,mem_free=500M launch_array_C.py
import os, sys

if __name__ == '__main__':

    # Cluster parameters.
    # queue_name = 'fast.master.q@compute-0-2'
    # python = '/home/1242538/virtualenvs/mlcv/bin/python'
    # python = '/usr/bin/python'
    # memory = '50M'
    # current_cmd = 'session3.py'

    cmd_mask = "python session3_provided.py"
    print cmd_mask
    print os.popen(cmd_mask).read()

#############################################################################