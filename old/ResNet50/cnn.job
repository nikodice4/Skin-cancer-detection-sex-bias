#!/bin/bash

#SBATCH --job-name=bsc-cnn-job    	# Job name
#SBATCH --output=job.cnn.%j.out  	# Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=8        	# Schedule 8 cores (includes hyperthreading)
#SBATCH --gres=gpu:1	          	# Schedule a GPU, or more with gpu:2 etc
#SBATCH --time=05:30:00          	# Run time (hh:mm:ss)
#SBATCH --partition=brown    		# Run on the Brown queue
#SBATCH --mail-type=FAIL,END          	# Send an email when the job finishes or fails

module load Anaconda3/2023.03-1

cd "/home/nizp/BSc-Project/ResNet50" 

source activate bachelor

SECONDS=0

echo "Running on $(hostname):"

python cnn_with_val.py

duration=$SECONDS
echo "All models took $(($duration / 60)) minutes and $(($duration % 60)) seconds to train."
