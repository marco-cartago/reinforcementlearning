import numpy as np
import time
import os
import subprocess
import matplotlib.pyplot as plt

from enviroment import GridWorld, GridWorldConfig
from agents import QLearning, Vapor
from tqdm import tqdm


CONFIG: GridWorldConfig = GridWorldConfig(
    size=5,
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
MAX_STEPS_PER_EPISODE = 15


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
    max_steps_per_episode = MAX_STEPS_PER_EPISODE
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
    show_final_path = True
    episode_rewards = []

    # Initialize VAPOR agent
    VAPOR_agent = Vapor(gridworld, terminal_states, horizon=MAX_STEPS_PER_EPISODE)

    for ep in tqdm(range(n_episodes)):
        # Reset environment
        gridworld = GridWorld(CONFIG)
        gridworld.reset()
        steps = 0
        total_reward = 0
        VAPOR_agent.gridworld = gridworld

        # Run episode
        while not gridworld.is_terminated and steps < MAX_STEPS_PER_EPISODE:
            s = gridworld.agent_pos
            a = VAPOR_agent.sample_action(steps, s)
            reward = gridworld.do_action(a)             # Take action
            total_reward += reward
            steps += 1
            if ep % 10 == 0:
                print(gridworld)
                time.sleep(0.01)

        # Learn and store episode reward
        episode = gridworld.get_episode()
        VAPOR_agent.learn_from_episode(episode)

        episode_rewards.append(total_reward)

    input("Press enter to continue...")

    # After training
    if show_final_path:
        clear_screen()
        print("\n=== Final Learned Path ===")

        # Reset environment for final demonstration
        gridworld = GridWorld(CONFIG)
        gridworld.reset()

        steps = 0
        while not gridworld.is_terminated and steps < MAX_STEPS_PER_EPISODE:
            s = gridworld.agent_pos
            a = VAPOR_agent.best_action(steps, s)
            reward = gridworld.do_action(a)
            steps += 1

            clear_screen()
            print(gridworld)
            time.sleep(2.0)  # Slow down for visualization

        print(f"\nFinal Path Reward: {gridworld.total_reward:.2f}")
        print(f"Steps taken: {steps}")

    print("Q-state -> lambda")
    for i in range(len(VAPOR_agent.legal_qstates)):
        if VAPOR_agent.curr_lambda[i] > 1e-7:
            print(f" - {VAPOR_agent.legal_qstates[i]} -> {VAPOR_agent.curr_lambda[i]}")

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
