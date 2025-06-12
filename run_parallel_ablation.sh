#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate cse276f

# Function to run a single experiment
run_experiment() {
    local gpu_id=$1
    local num_hidden=$2
    local trial_num=$3
    
    # Set GPU device
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Calculate seed based on trial number (using different seeds for each trial)
    local seed=$((42 + trial_num * 1000 + num_hidden))
    
    # Run the experiment
    python -m examples.baselines.ppo.ppo \
        --env_id="PlaceBananaInBin-v1" \
        --num_envs=2048 \
        --update_epochs=8 \
        --num_minibatches=32 \
        --total_timesteps=200_000_000 \
        --eval_freq=10 \
        --num_steps=50 \
        --exp_name="banana_${num_hidden}_layers_trial_${trial_num}" \
        --num_hidden=$num_hidden \
        --seed=$seed
}

# Number of trials per configuration
NUM_TRIALS=5

# Launch experiments for each hidden layer configuration
# Each configuration runs on its own GPU
for hidden in 1 2 3 4 5; do
    for trial in $(seq 0 $((NUM_TRIALS-1))); do
        # Run in background
        run_experiment $((hidden-1)) $hidden $trial &
    done
done

# Wait for all background processes to complete
wait 
