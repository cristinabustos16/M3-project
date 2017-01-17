# It is intended to be called with the following command:
# qsub -cwd -V -S /home/1442514/virtualenvs/mlcv/bin/python -q fast.master.q@compute-0-2 -l mem_token=1000M,mem_free=1000M lanza1.py
import os

if __name__ == '__main__':

    cmd = "python session3_provided.py"
    print cmd
    print os.popen(cmd).read()

#############################################################################