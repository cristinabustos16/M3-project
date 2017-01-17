# Program to try different numbers of principal components in parallel.
# It is intended to be called like an array job, with the following command:
# qsub -cwd -V -S /usr/bin/python -q fast.master.q@compute-0-2 -t 1-10 -l mem_token=2000M,mem_free=2000M launch_array_pca_fisher.py
import os, sys


if __name__ == '__main__':
    
    # Other parameters are specified in try_SVM_full.py

    # Range of parameters to try:
    ncomp_pca_vec = (1, 2, 3, 4, 5, 6, 7, 10, 20, 30, 40, 50, 60, 70, \
                    80, 90, 100, 110, 120, 128)
                    
    # Get the environment variables .............................
    task_id = int(os.environ['SGE_TASK_ID'])
    task_first = int(os.environ['SGE_TASK_FIRST'])
    task_end = int(os.environ['SGE_TASK_LAST']) + 1
    task_step = int(os.environ['SGE_TASK_STEPSIZE'])
    
    # Get the task ID ...........................................
    tasks = range(task_first, task_end, task_step)
    number_of_jobs = len(tasks)
    job_id = tasks.index(task_id)
    
    cmd_mask = "python try_PCA_full_fisher.py %s %s"
    for index in range(job_id, len(ncomp_pca_vec), number_of_jobs):
        ncomp_pca = ncomp_pca_vec[index]
        resfile = 'res_PCA_' + str(ncomp_pca) + '.txt'
        cmd = cmd_mask % (resfile, ncomp_pca)
        print cmd
        print os.popen(cmd).read()
        sys.stdout.flush()
        
        
        
        
        
        