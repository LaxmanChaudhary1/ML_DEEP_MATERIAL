#!/bin/bash
#SBATCH -J xbgm             # Job name
#SBATCH -p normal       # Partition name
#SBATCH -o job.%j.out   # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1               # Total number of nodes requested
#SBATCH -n 20               # Total number of mpi tasks requested
#SBATCH -t infinite       # Run time (hh:mm:ss)#!/bin/bash

#Launch MPI-based executable


#python /home/mcsitgb/storage0/xgboost/parameterTuner.py > out_tunner
#python /home/mcsitgb/storage0/xgboost/xgbrgrid.py > out_grid
#python /home/mcsitgb/storage0/xgboost/xgbrmodel.py > out_model_tuned

