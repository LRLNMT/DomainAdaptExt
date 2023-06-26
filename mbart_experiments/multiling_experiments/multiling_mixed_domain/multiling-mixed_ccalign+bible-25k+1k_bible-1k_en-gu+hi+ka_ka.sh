#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-user=cindyy.huang@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.8
source /home/huan1766/scratch/env/bin/activate

pip install --no-index --upgrade pip

module load gcc/9.3.0
module load cuda/11.4
module load arrow/8.0.0

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
bash mixed_domain_o2m_stage2_single.sh 3 0 Gujarati Hindi Kannada cc_aligned+new_bible_g1 25k+1k Kannada new_bible_g1 1k
