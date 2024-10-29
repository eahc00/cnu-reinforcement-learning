import numpy as np
import gymnasium as gym

""" 
Cliff-walking-v0
- Observation space
  . Type: gym.spaces.Discrete(48)
  . Goal state: 47
  . Cliff states: 37 ~ 46
- Action space
  . Type: gym.spaces.Discrete(4)
  . Actions: 0(Up), 1(Right), 2(Down), 3(Left)
- Reward
  . Step reward: -1
  . Cliff panalty: -100
  . Goal reward: 0
"""

def run_environment(env, policy=None, n_iterations=1):
    """
    This function runs a Gym environment
    by following a policy or sampling actions randomly from the action_space.

    Args:
        env: Gymnasium environment
        policy: Policy to follow while playing an episode; if None, uses a random policy
        n_iterations: Number of episodes to run

    Returns:
        n_success: Total number of successes out of n_iterations
        average_reward: Average reward over n_iterations
    """
    # Initialize success count and total reward
    n_success = 0
    total_reward = 0

    # Loop over the number of episodes to play
    for i in range(n_iterations):
        # Reset the environment every time when starting a new episode
        state, _ = env.reset()

        # Initialize variables for the episode
        episode_reward = 0
        n_steps = 0  
        while True:
            # If the policy is not given, take a random action; else, take an action according to the policy
            if policy is None:
                action = env.action_space.sample()
            else:
                action = policy[state]

            # Take the next step
            next_state, reward, terminated, truncated, info = env.step(action)
            n_steps += 1  # Increment step counter

            # Update episode reward
            episode_reward += reward

            # Check if the episode is successfully over
            if terminated:
                n_success += 1
                print("success {}/{}".format(i+1, n_iterations))
                print("Episode {}: Step {}, Episode_rewards {}".format(i+1, n_steps, episode_reward))
                break  # End the episode

            # Update the state
            state = next_state

        # Update total rewards
        total_reward += episode_reward

    # Calculate average reward
    average_reward = total_reward / n_iterations

    return n_success, average_reward



if __name__ == "__main__":   
    env_name = 'CliffWalking-v0'
    training_env = gym.make(env_name, render_mode="rgb_array")
    # Q, policy = q_learning(training_env, alpha = 0.5, gamma = 0.98, n_episodes = 3000)
    # training_env.close()


    # test_env = gym.make(env_name, render_mode="human")
    # run_environment(test_env, policy=policy, n_iterations=5)
    # test_env.close()


    env = gym.make(env_name, render_mode="human")
    run_environment(env, policy=None, n_iterations=10)
    env.close()        
