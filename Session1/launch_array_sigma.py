# Program to try many different sigmas for SVM in parallel.
# It is intended to be called like an array job, with the following command:
# qsub -cwd -V -S /usr/bin/python -q fast.master.q@compute-0-2 -t 1-10 -l mem_token=500M,mem_free=500M launch_array.py
import os, sys


if __name__ == '__main__':
    
    # Fixed parameters:
    SVM_kernel = 'rbf'
    SVM_C = 1
    
    # Other parameters are specified in try_SVM_full.py

    # Range of parameters to try:
    SVM_sigma_vec = (0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.02, 0.03, \
            0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
                    
    # Get the environment variables .............................
    task_id = int(os.environ['SGE_TASK_ID'])
    task_first = int(os.environ['SGE_TASK_FIRST'])
    task_end = int(os.environ['SGE_TASK_LAST']) + 1
    task_step = int(os.environ['SGE_TASK_STEPSIZE'])
    
    # Get the task ID ...........................................
    tasks = range(task_first, task_end, task_step)
    number_of_jobs = len(tasks)
    job_id = tasks.index(task_id)
    
    cmd_mask = "python try_SVM_full.py %s %s %s %s"
    for index in range(job_id, len(SVM_sigma_vec), number_of_jobs):
        SVM_sigma = SVM_sigma_vec[index]
        resfile = 'res_sigma_' + str(SVM_sigma) + '.txt'
        cmd = cmd_mask % (resfile, SVM_kernel, str(SVM_C), SVM_sigma)
        print cmd
        print os.popen(cmd).read()
        sys.stdout.flush()
        
        
        
        
        
        