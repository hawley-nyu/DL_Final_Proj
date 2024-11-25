srun --account=csci_ga_2572-2024fa --partition=c12m85-a100-1 --gres=gpu --time=01:00:00 --pty /bin/bash

singularity exec --bind /scratch --nv --overlay /scratch/zg915/overlay-25GB-500K.ext3:rw /scratch/zg915/ubuntu-20.04.3.sif /bin/bash

source /ext3/miniconda3/etc/profile.d/conda.sh

conda activate my_env