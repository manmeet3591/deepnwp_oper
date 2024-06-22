Apptainer_sfno.def  min_max.csv  README.md  src  submit.sh
login1.ls6(1054)$ vim submit.sh 

#!/bin/bash
#SBATCH --job-name=gpu_apptainer               # Name of the job
#SBATCH --ntasks=1                          # Number of tasks
#SBATCH --nodes=1                           # Ensure that all cores are on one machine
#SBATCH --time=0-48:00                      # Runtime in D-HH:MM
#SBATCH --output=out.%j                     # File to which STDOUT will be written
#SBATCH --error=err.%j                      # File to which STDERR will be written
#SBATCH --mail-type=ALL                     # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=manmeet.singh@utexas.edu                    # Email to which notifications will be sent
#SBATCH -p gpu-a100-small                              # Partition to submit to



module load tacc-apptainer

apptainer instance start --nv --bind /scratch/08105/ms86336:/workspace sfno_v3.sif apptainer_instance

sleep infinity
