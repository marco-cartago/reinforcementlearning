import numpy as np
import time
import os
import subprocess
import matplotlib.pyplot as plt

from enviroment import GridWorld, GridWorldConfig
from agents import QLearning, VAPOR
from tqdm import tqdm


CONFIG: GridWorldConfig = GridWorldConfig(
    size=4,
    p_walls=0.6,
    agent_start=np.array((0, 0)),
    step_penalty=-(2 ** (-10)),
    small_treasure_rew=1,
    treasure_rew=100,
    sd_small_treasure=1.0,
    sd_treasure=1.0,
    temperature=0.1,
    gamma=0.995,
    random_state=0
)


def clear_screen():
    command = "cls" if os.name == "nt" else "clear"
    subprocess.run([command], shell=True)


def main_QLEARNING():
    # Initialize the environment
    gridworld = GridWorld(CONFIG)

    # Terminal states are the treasure positions
    terminal_states = [gridworld.treasure_pos, gridworld.small_treasure_pos]

    # Initialize Q-learning agent
    q_agent = QLearning(gridworld, terminal_states, alpha=1)
    n_episodes = 20_000
    max_steps_per_episode = 1_000
    show_final_path = True
    episode_rewards = []

    for episode in tqdm(range(n_episodes)):
        # Reset environment
        gridworld = GridWorld(CONFIG)
        gridworld.reset()
        steps = 0
        total_reward = 0
        q_agent.gridworld = gridworld
        q_agent.alpha = 0.1 * (1 - episode / n_episodes)  # Decaying learning rate

        # Run episode
        while not gridworld.is_terminated and steps < max_steps_per_episode:
            s = gridworld.agent_pos
            a = q_agent.best_action_epsilon_greedy(s, epsilon=0.5)
            # Take action
            reward = gridworld.do_action(a)
            total_reward += reward
            steps += 1
            if episode % 10_000 == 0:
                print(gridworld)
                time.sleep(0.01)

        # Learn
        q_agent.learn_from_episode()

        # Store episode reward
        episode_rewards.append(total_reward)


    # After training
    if show_final_path:
        clear_screen()
        print("\n=== Final Learned Path ===")

        # Reset environment for final demonstration
        gridworld = GridWorld(CONFIG)
        gridworld.reset()
        steps = 0

        while not gridworld.is_terminated and steps < max_steps_per_episode:
            s = gridworld.agent_pos
            a = q_agent.best_action(s)
            reward = gridworld.do_action(a)
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


def main_VAPOR():
    # Initialize the environment
    gridworld = GridWorld(CONFIG)

    # Terminal states are the treasure positions
    terminal_states = [gridworld.treasure_pos, gridworld.small_treasure_pos]
    n_episodes = 100
    max_steps_per_episode = 20
    show_final_path = True
    episode_rewards = []

    # Initialize VAPOR agent
    VAPOR_agent = VAPOR(gridworld, terminal_states, horizon=max_steps_per_episode)

    for episode in tqdm(range(n_episodes)):
        # Reset environment
        gridworld = GridWorld(CONFIG)
        gridworld.reset()
        steps = 0
        total_reward = 0
        VAPOR_agent.gridworld = gridworld

        # Run episode
        while not gridworld.is_terminated and steps < max_steps_per_episode:
            s = gridworld.agent_pos
            a = VAPOR_agent.best_action(steps, s)
            reward = gridworld.do_action(a)             # Take action
            total_reward += reward
            steps += 1
            if episode % 100 == 0:
                # print(gridworld)
                time.sleep(0.01)

        # Learn and store episode reward
        episode = gridworld.get_episode()
        VAPOR_agent.learn_from_episode(episode)

        episode_rewards.append(total_reward)

    # After training
    if show_final_path:
        clear_screen()
        print("\n=== Final Learned Path ===")

        # Reset environment for final demonstration
        gridworld = GridWorld(CONFIG)
        gridworld.reset()

        steps = 0
        while not gridworld.is_terminated and steps < max_steps_per_episode:
            s = gridworld.agent_pos
            a = VAPOR_agent.best_action(steps, s)

            reward = gridworld.do_action(a)
            steps += 1
            clear_screen()
            print(gridworld)
            time.sleep(0.05)  # Slow down for visualization

        print(f"\nFinal Path Reward: {gridworld.total_reward:.2f}")
        print(f"Steps taken: {steps}")

    for l in VAPOR_agent.table_lambda.keys():
        print(l, VAPOR_agent.table_lambda[l])

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    clear_screen()
    # main_QLEARNING()
    main_VAPOR()
