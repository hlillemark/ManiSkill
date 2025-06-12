import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob
import re
from collections import defaultdict

def find_tensorboard_file(log_dir):
    """Find the tensorboard event file in the given directory."""
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No tensorboard event files found in {log_dir}")
    return event_files[0]

def load_tensorboard_data(log_dir, tags):
    """Load data from tensorboard logs."""
    event_file = find_tensorboard_file(log_dir)
    print(f"Found tensorboard file: {event_file}")
    
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    
    data = {}
    for tag in tags:
        if tag in ea.scalars.Keys():
            events = ea.scalars.Items(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            data[tag] = (steps, values)
    return data

def find_experiment_runs(base_dir):
    """Find all experiment runs matching the pattern banana_*_layers_trial_*."""
    pattern = re.compile(r'banana_(\d+)_layers_trial_(\d+)')
    runs = defaultdict(list)
    
    for item in os.listdir(base_dir):
        match = pattern.match(item)
        if match:
            num_hidden = int(match.group(1))
            trial_num = int(match.group(2))
            runs[num_hidden].append((trial_num, os.path.join(base_dir, item)))
    
    # Sort by trial number for each configuration
    for num_hidden in runs:
        runs[num_hidden].sort(key=lambda x: x[0])
        runs[num_hidden] = [path for _, path in runs[num_hidden]]
    
    return runs

def aggregate_experiment_data(runs_dict, tags):
    """Aggregate data from multiple runs of the same configuration."""
    aggregated_data = {}
    
    for num_hidden, run_paths in runs_dict.items():
        print(f"\nProcessing configuration with {num_hidden} hidden layers")
        print(f"Found {len(run_paths)} trials")
        
        all_data = defaultdict(list)
        
        for run_path in run_paths:
            try:
                data = load_tensorboard_data(run_path, tags)
                for tag in tags:
                    if tag in data:
                        steps, values = data[tag]
                        all_data[tag].append((steps, values))
            except Exception as e:
                print(f"Error loading data from {run_path}: {e}")
                continue
        
        # Process aggregated data
        processed_data = {}
        for tag in tags:
            if tag in all_data:
                # Get all unique steps across all runs
                all_steps = set()
                for steps, _ in all_data[tag]:
                    all_steps.update(steps)
                all_steps = sorted(list(all_steps))
                
                # Interpolate values for each run to match the common steps
                interpolated_values = []
                for run_steps, run_values in all_data[tag]:
                    interpolated = np.interp(all_steps, run_steps, run_values)
                    interpolated_values.append(interpolated)
                
                if interpolated_values:
                    mean_values = np.mean(interpolated_values, axis=0)
                    # Calculate standard error of the mean (SEM) instead of std
                    sem_values = np.std(interpolated_values, axis=0) / np.sqrt(len(interpolated_values))
                    processed_data[tag] = (all_steps, mean_values, sem_values)
                    print(f"Successfully processed {len(interpolated_values)} trials for {tag}")
        
        aggregated_data[num_hidden] = processed_data
    
    return aggregated_data

def plot_aggregated_curves(aggregated_data, save_path=None):
    """Plot aggregated training curves with error bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for different configurations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot success rate
    for i, (num_hidden, data) in enumerate(sorted(aggregated_data.items())):
        if 'eval/success_once' in data:
            steps, mean_values, sem_values = data['eval/success_once']
            ax1.plot(steps, mean_values, color=colors[i], label=f'{num_hidden} Hidden Layers')
            # Clip error bars to [0, 1] for success rate
            lower_bound = np.clip(mean_values - sem_values, 0, 1)
            upper_bound = np.clip(mean_values + sem_values, 0, 1)
            ax1.fill_between(steps, 
                           lower_bound,
                           upper_bound,
                           color=colors[i], alpha=0.2)
    
    ax1.set_xlabel('Environment Steps')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate Over Training')
    ax1.grid(True)
    ax1.legend()
    
    # Plot return
    for i, (num_hidden, data) in enumerate(sorted(aggregated_data.items())):
        if 'eval/return' in data:
            steps, mean_values, sem_values = data['eval/return']
            ax2.plot(steps, mean_values, color=colors[i], label=f'{num_hidden} Hidden Layers')
            ax2.fill_between(steps,
                           mean_values - sem_values,
                           mean_values + sem_values,
                           color=colors[i], alpha=0.2)
    
    ax2.set_xlabel('Environment Steps')
    ax2.set_ylabel('Return')
    ax2.set_title('Return Over Training')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Base directory containing experiment runs")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the figure (optional)")
    args = parser.parse_args()
    
    # Find all experiment runs
    runs_dict = find_experiment_runs(args.log_dir)
    print(f"Found runs for configurations: {sorted(runs_dict.keys())}")
    
    # Aggregate data
    aggregated_data = aggregate_experiment_data(runs_dict, ['eval/success_once', 'eval/return'])
    
    # Plot results
    plot_aggregated_curves(aggregated_data, args.save_path) 