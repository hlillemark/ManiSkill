import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob

def find_tensorboard_file(log_dir):
    """Find the tensorboard event file in the given directory."""
    # Look for files starting with events.out.tfevents
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No tensorboard event files found in {log_dir}")
    return event_files[0]  # Return the first event file found

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

def plot_training_curves(log_dir, save_path=None):
    """Plot success rate and return curves."""
    # Load data
    data = load_tensorboard_data(
        log_dir,
        tags=['eval/success_once', 'eval/return']
    )
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot success rate
    if 'eval/success_once' in data:
        steps, values = data['eval/success_once']
        ax1.plot(steps, values, 'b-', label='Success Rate')
        ax1.set_xlabel('Environment Steps')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate Over Training')
        ax1.grid(True)
        ax1.legend()
        
        # Add moving average
        window_size = 5
        if len(values) > window_size:
            moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(steps[window_size-1:], moving_avg, 'r--', label=f'{window_size}-step Moving Average')
            ax1.legend()
    
    # Plot return
    if 'eval/return' in data:
        steps, values = data['eval/return']
        ax2.plot(steps, values, 'g-', label='Return')
        ax2.set_xlabel('Environment Steps')
        ax2.set_ylabel('Return')
        ax2.set_title('Return Over Training')
        ax2.grid(True)
        ax2.legend()
        
        # Add moving average
        window_size = 5
        if len(values) > window_size:
            moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(steps[window_size-1:], moving_avg, 'r--', label=f'{window_size}-step Moving Average')
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
    parser.add_argument("--log_dir", type=str, required=True, help="Path to tensorboard log directory")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the figure (optional)")
    args = parser.parse_args()
    
    plot_training_curves(args.log_dir, args.save_path) 
