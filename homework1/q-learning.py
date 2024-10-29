import gymnasium as gym
import numpy as np

from cliff_walking import run_environment


def Q_learning(env, alpha=0.8, gamma=0.98, epsilon=1.0, epsilon_decay=0.001, n_episodes=1000, save_path="q_tables"):
    """
    Q-learning algorithm
        Q(s,a) <- Q(s,a) + alpha*(reward + gamma*max_aQ(s',a) - Q(s,a))
    
    Args:
        env: openAI Gym environment
        alpha: learning rate (0~1)
        gamma: discount factor
        epsilon: probability to select a random action
        epsilon_decay: parameters to decay epsilon
        n_episodes: number of episodes to run    
    Return:
        Q: action-value function
        optimal_policy: optimal policy
    """    

    # Algorithm parameters ----------------------------------------------------------
    #   set function arguments
    # alpha = 0.8               # learning rate
    # gamma = 0.98              # discount factor
    # epsilon = 1.0             # starting value of epsilon-greedy probability
    # epsilon_decay = 0.001     # epsilon decay rate
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Initialize Q(s,a) for all s, a except that Q(terminal,.)=0 --------------------
    # -------------------------------------------------------------------------------
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))


    # -------------------------------------------------------------------------------
    # Loop for each episide ---------------------------------------------------------
    # -------------------------------------------------------------------------------
    for episode in range(n_episodes):    
        # Initialize S --------------------------------------------------------------
        state, _ = env.reset()

        # Choose A from S using policy derived from Q with e-greedy -----------------
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        
        # Loop for each step of episode ----------------------------------------------
        episode_rewards = 0
        while True:        
            # Take an action A, observe R, S' ---------------------------------------- 
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_rewards += reward
            
            # Choose A' from S' using policy derived from Q with Max ------------
            next_action = np.argmax(Q[next_state, :])
            
            # Q(s,a) <- Q(s,a) + alpha*(reward + gamma*Q(s',a) - Q(s,a)) -------------
            td_target = reward + gamma * Q[next_state, next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] = Q[state, action] + alpha * td_error
                    
            # S <- S', A -> A' -------------------------------------------------------
            state = next_state
            action = next_action

            # until S is terminal ------------------------------------------
            if terminated or truncated:
                break
            
        # decay epsilon
        epsilon = max(epsilon - epsilon_decay, 0)

        # print training status
        if (episode + 1) % 10 == 0:
            print("Episode {}/{}, Episode rewards: {}".format(episode+1, n_episodes, episode_rewards))

    # # save Q function (q_table)
    # if save_path:
    #    os.makedirs(save_path, exist_ok=True)
    #    with open("{}/frozenLake_sarsa.pkl".format(save_path), 'wb') as f:
    #        pickle.dump(Q, f)

    # Deterministic optimal policy
    optimal_policy = np.argmax(Q, axis=1)
 
    return Q, optimal_policy
 

if __name__ == "__main__":
    
    env_name = 'CliffWalking-v0'
    training_env = gym.make(env_name, is_slippery=False)
    
    alpha=0.8
    gamma = 0.98
    epsilon = 1.0
    epsilon_decay = 0.001  
    n_episodes = 1000
    save_path = "q_tables"
    Q, policy = Q_learning(
        training_env, 
        alpha=alpha, 
        gamma=gamma, 
        epsilon=epsilon,  
        epsilon_decay=epsilon_decay, 
        n_episodes=n_episodes, 
        save_path=save_path, 
    )
    print("Training Done -----------------------------------")
    print("Optimal Policy:")
    print(policy)
    training_env.close()
    
    # with open("q_tables/frozenLake_sarsa.pkl", 'rb') as f:
    #   Q = pickle.load(f)
    # policy = np.argmax(Q, axis=1)

    test_env = gym.make(env_name, is_slippery=False, render_mode="human")
    n_success, average_reward = run_environment(test_env, policy, n_iterations=1)
    print("Test Done ---------------------------------------")
    print("number of success: {}".format(n_success))
    print("average reward: {}".format(average_reward))
    test_env.close()

