# It is intended to be called with the following command:
# qsub -cwd -V -S /home/1442514/virtualenvs/mlcv/bin/python -q fast.master.q@compute-0-2 -l mem_token=1000M,mem_free=1000M lanza3.py
import os

if __name__ == '__main__':

    cmd = "python cross_validation_fisher2.py"
    print cmd
    print os.popen(cmd).read()

#############################################################################