#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
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
bash mixed_domain_fairseq_stage1.sh Sinhala 0 cc_aligned+government+new_bible_g2 25k+25k+25k new_bible_g2