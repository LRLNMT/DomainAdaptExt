#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-user=xin.peng@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load anaconda3
#conda create -n pytorch_env python=3.8
source activate pytorch_env

#conda config --prepend channels https://ftp.osuosl.org/pub/open-ce/1.7.2/
#conda config --set channel_priority strict
#conda install -c https://ftp.osuosl.org/pub/open-ce/1.7.2/ pytorch=1.12.1 cudatoolkit=11.4

#conda clean -y --all
#rm -rf $HOME/.conda/pkgs/*
#rm -f $HOME/.condarc

module load cuda/11.4.4
module load gcc/11.3.0
module load arrow/4.0.1

#git clone https://github.com/arianajung/fairseq
#cd fairseq
#pip install --upgrade pip
#pip install --editable ./

#cd ..
#pip install sentencepiece sacrebleu tensorboardX

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
bash mixed_domain_fairseq_stage_2.sh Gujarati 0 cc_aligned+PrimeMinisterCorpus 25k+25k PrimeMinisterCorpus 25k
