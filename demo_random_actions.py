import gymnasium as gym
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode


# Create environment with video recording
env = gym.make(
    "PlaceBananaInBin-v1",
    num_envs=1,
    obs_mode="state",
    render_mode="rgb_array"
)
env = RecordEpisode(
    env,
    output_dir="./random_actions",
    save_trajectory=False,
    save_video_trigger=lambda x: True,  # Save every episode
    max_steps_per_video=50,
    video_fps=30,
)
# due to how RecordEpisode works, it will save the video at ./random_actions/0.mp4

# Run one episode with random actions
obs, _ = env.reset()
done = False
truncated = False

while not (done or truncated):
    # Sample random action from action space and then take step
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

env.close() 
