import numpy as np
import time
import os
from enviroment import GridWorld
from agents import QLearning

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # Initialize the environment
    gridworld = GridWorld(
        size=13, 
        p_walls=0.2,
        agent_start=(0, 0),
        step_penalty=-(2**(-10)),
        small_treasure_rew=10,
        treasure_rew=1_000,
        temperature=0.01,
        gamma=0.9, 
        random_state = np.random.RandomState(0)
    )

    # Terminal states are the treasure positions
    terminal_states = [gridworld.treasure_pos, gridworld.small_treasure_pos]

    # Initialize Q-learning agent
    q_agent = QLearning(gridworld, terminal_states, alpha=0.1)

    n_episodes = 100_000
    max_steps_per_episode = 1000
    show_final_path = True
    episode_rewards = []

    for episode in range(n_episodes):
        # Reset environment
        gridworld = GridWorld(
            size=gridworld.size,
            p_walls=gridworld.p_walls,
            agent_start=gridworld.agent_start,
            step_penalty=gridworld.step_penalty,
            small_treasure_rew=gridworld.small_treasure_rew,
            treasure_rew=gridworld.treasure_rew,
            sd_treasure=gridworld.sd_treasure,
            sd_small_treasure=gridworld.sd_small_treasure,
            temperature=gridworld.temperature,
            gamma=gridworld.gamma,
            random_state=np.random.RandomState(0)
        )
        gridworld.current_episode = []  # Clear previous episode
        gridworld.is_terminated = False
        gridworld.total_reward = 0
        gridworld.step = 0
        total_reward = 0
        steps = 0

        # Run episode
        while not gridworld.is_terminated and steps < max_steps_per_episode:
            s = gridworld.agent_pos

            # Choose action
            if np.random.rand() < 0.2:
                # Explore
                actions = gridworld.get_actions()
                a_idx = np.random.randint(0, len(actions))
                a = actions[a_idx]
            else:
                # Exploit
                a = q_agent.best_action(s)
                if a is None:
                    actions = gridworld.get_actions()
                    a_idx = np.random.randint(0, len(actions))
                    a = actions[a_idx]
                else:
                    a_idx = gridworld.get_actions().index(a) if a in gridworld.get_actions() else 0

            # Take action
            reward = gridworld.do_action(gridworld.get_actions(), a_idx)
            total_reward += reward

            steps += 1
            if steps % 10 == 0 and episode % 100 == 0:
                print(gridworld)
                time.sleep(0.001)

        # Learn
        q_agent.learn_from_episode()

        # Store episode reward
        episode_rewards.append(total_reward)

        # Print progress
        #print(f"Episode {episode + 1}/{n_episodes} | Reward: {total_reward:.2f} | Steps: {steps}\n")

    #
    # After training
    # 

    if show_final_path:
        clear_screen()
        print("\n=== Final Learned Path ===")

        # Reset environment for final demonstration
        gridworld.__init__(
            size=gridworld.size,
            p_walls=gridworld.p_walls,
            agent_start=gridworld.agent_start,
            step_penalty=gridworld.step_penalty,
            small_treasure_rew=gridworld.small_treasure_rew,
            treasure_rew=gridworld.treasure_rew,
            sd_treasure=gridworld.sd_treasure,
            sd_small_treasure=gridworld.sd_small_treasure,
            temperature=gridworld.temperature,
            gamma=gridworld.gamma
        )
        gridworld.current_episode = []
        gridworld.is_terminated = False
        gridworld.total_reward = 0
        gridworld.step = 0

        steps = 0
        while not gridworld.is_terminated and steps < max_steps_per_episode:
            s = gridworld.agent_pos
            a = q_agent.best_action(s)

            if a is None:
                actions = gridworld.get_actions()
                a_idx = np.random.randint(0, len(actions))
            else:
                a_idx = gridworld.get_actions().index(a) if a in gridworld.get_actions() else 0

            reward = gridworld.do_action(gridworld.get_actions(), a_idx)
            steps += 1
            clear_screen()
            print(gridworld)
            time.sleep(0.1)  # Slow down for visualization

        print(f"\nFinal Path Reward: {gridworld.total_reward:.2f}")
        print(f"Steps taken: {steps}")

    # Plot rewards
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()