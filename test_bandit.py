import matplotlib.pyplot as plt
import numpy as np
from bandit_agents import SoftQLearningBandit, VaporBandit, QLearningBandit, a2idx
from enviroment import MultiArmedBandit

def main_VAPOR_bandit():
    MAB_SIZE = 10            
    VAR_OF_MU = 1.0          
    SHAPE = 1.0              
    SCALE = 1.0             
    N_EPISODES = 10_000
    
    bandit = MultiArmedBandit(
        size=MAB_SIZE, 
        var_of_mu=VAR_OF_MU, 
        shape=SHAPE, 
        scale=SCALE
    )
    episode_rewards = []

    agent = VaporBandit(bandit)
    print(bandit)

    for ep in range(N_EPISODES):
        # Reset environment
        bandit.reset()
        s = bandit.agent_pos 
        a = agent.get_action(l=0, s=0)
        reward = bandit.do_action(a)
        agent.learn_from_episode()
        
        episode_rewards.append(reward)
        
        if ep % 1000 == 0:
            print(f"Episode={ep} Reward={reward:.2f}")

    print("Training complete.")

    actual_best_arm = np.argmax(bandit.mu)
    predicted_best_arm = a2idx(agent.get_action(0)) 
    
    print(f"\nActual best arm (highest mu): {actual_best_arm}")
    print(f"Agent's predicted best arm: {predicted_best_arm}")
    
    if actual_best_arm == predicted_best_arm:
        print("Success: The agent found the optimal arm!")
    else:
        print("Failure: The agent converged to a sub-optimal arm.")

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Reward per Episode (Multi-Armed Bandit)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main_VAPOR_bandit()
