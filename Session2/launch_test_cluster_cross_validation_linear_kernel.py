# Program to try many different costs for SVM in parallel.
# It is intended to be called like an array job, with the following command:
# qsub -cwd -V -S /usr/bin/python -q fast.master.q@compute-0-2 -t 1-10 -l mem_token=500M,mem_free=500M launch_array_C.py
import os, sys

if __name__ == '__main__':

    # Fixed parameters:
    param1 = 'test_linear_sift_kmeans_'
    param2 = 'linear'     # kernel
    param3 = 'SIFT'       # descriptor
    param4 = 100          # features
    param5 = 0            # apply dense sampling
    param6 = 0            # apply spatial pyramids
    param7 = 3            # spatial pyramids depth

    # Range of parameters to try (param8):
    Kmeans_K = (512, 1024, 2048, 9056)

    # Get the environment variables .............................
    task_id = int(os.environ['SGE_TASK_ID'])
    task_first = int(os.environ['SGE_TASK_FIRST'])
    task_end = int(os.environ['SGE_TASK_LAST']) + 1
    task_step = int(os.environ['SGE_TASK_STEPSIZE'])

    # Get the task ID ...........................................
    tasks = range(task_first, task_end, task_step)
    number_of_jobs = len(tasks)
    job_id = tasks.index(task_id)

    cmd_mask = "python session2_cross_validation_train_linear_kernel.py %s %s %s %s %s %s %s %s"
    for index in range(job_id, len(Kmeans_K), number_of_jobs):
        param8 = Kmeans_K[index]
        report_filename = param1 + str(param8) + '.txt'
        cmd = cmd_mask % (report_filename, param2, param3, str(param4),str(param5),str(param6), str(param7), str(param8))
        print cmd
        print os.popen(cmd).read()
        sys.stdout.flush()

#############################################################################