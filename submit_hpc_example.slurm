#!/bin/bash
#----------------------------------------------------
#SBATCH -J Prandtl_InletBuzz    	# Job Name
#SBATCH -o Prandtl_InletBuzz.o%j	# Output File
#SBATCH -e Prandtl_InletBuzz.e%j	# Error File
#SBATCH -p spr				        # Queue (partition) name
#SBATCH -N 1				        # Nodes
#SBATCH -n 112				        # Cores
#SBATCH -t 02:00:00			        # Time
#SBATCH -A TG-CTS090004			    # Your ID

# --- Load Modules ---
module load intel/25.1
module load cmake/3.31.5

# --- Change to the run directory on SCRATCH ---
cd $SCRATCH/Prandtl_Runs

# --- Set Run Parameters ---
export OMP_NUM_THREADS=1
CONFIG_FILE="Nagashima_Ramjet/TR0/config.json"
N_PROCS=112

# --- Run the simulation using ibrun ---
echo "Starting Prandtl simulation in $(pwd)"
ibrun ./Prandtl -c "$CONFIG_FILE" -np ${N_PROCS}
echo "Simulation finished."
