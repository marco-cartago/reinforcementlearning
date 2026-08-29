import numpy as np
import time
import os
import subprocess
import matplotlib.pyplot as plt

from utils import a2idx
from enviroment import GridWorld
from utils import GridWorldConfig
from agents import QLearning, Vapor, SoftQLearning
from tqdm import tqdm

SIZE = 8
CONFIG: GridWorldConfig = GridWorldConfig(
    size=SIZE,
    p_walls=0.70,
    agent_start=np.array((0, 0)),
    step_penalty=-(2 ** (-10)),
    small_treasure_rew=1e-3,
    treasure_rew=1,
    sd_small_treasure=1e-3,
    sd_treasure=1e-3,
    temperature=0.0,
    gamma=0.995,
    random_state=4
)
MAX_STEPS_PER_EPISODE = int(3 * SIZE)
N_EPISODES = 100


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
    n_episodes = N_EPISODES
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
            if episode % 10 == 0:
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
    #plt.savefig(f"./figures/run_q_{time.time_ns()}")


def main_VAPOR():
    # Initialize the environment
    gridworld = GridWorld(CONFIG)

    # Terminal states are the treasure positions
    terminal_states = [a2idx(gridworld.treasure_pos), a2idx(gridworld.small_treasure_pos)]
    n_episodes = N_EPISODES
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
            a = VAPOR_agent.sample_action(steps, s, eps=1e-8)
            reward = gridworld.do_action(a)             # Take action
            total_reward += reward
            steps += 1
            if ep % 10 == 0:
                print(gridworld)
                time.sleep(0.1)

        # Learn and store episode reward
        episode = gridworld.get_episode()
        VAPOR_agent.learn_from_episode(episode)

        episode_rewards.append(total_reward)

    # input("Press enter to continue...")

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

            time.sleep(1.0)  # Slow down for visualization
            clear_screen()
            print(gridworld)

        print(f"\nFinal Path Reward: {gridworld.total_reward:.2f}")
        print(f"Steps taken: {steps}")

    # print("Q-state -> lambda")
    # for i in range(len(VAPOR_agent.legal_qstates)):
    #     print(f" - {VAPOR_agent.legal_qstates[i]} -> {VAPOR_agent.curr_lambda[i]}")

    # print("Q-state -> Er")
    # for i in range(len(VAPOR_agent.legal_qstates)):
    #     print(f" - {VAPOR_agent.legal_qstates[i]} -> {VAPOR_agent.curr_reward_mean[i]}")

    # print("Q-state -> Var")
    # for i in range(len(VAPOR_agent.legal_qstates)):
    #     print(f" - {VAPOR_agent.legal_qstates[i]} -> {VAPOR_agent.curr_reward_variance[i]}")


    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    #plt.savefig(f"./figures/run_vapor_{time.time_ns()}")



def main_SoftQLEARNING():
    # Initialize the environment
    gridworld = GridWorld(CONFIG)

    # Terminal states are the treasure positions
    terminal_states = [gridworld.treasure_pos, gridworld.small_treasure_pos]

    temp_start = 1.3
    temp_end = 0.05

    # Initialize Q-learning agent
    soft_q_agent = SoftQLearning(gridworld, terminal_states, alpha=1, temperature=temp_start)
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
        soft_q_agent.gridworld = gridworld
        soft_q_agent.alpha = 0.2 * (1 - episode / n_episodes)  # Decaying learning rate
        if episode > n_episodes/2:
            soft_q_agent.temperature = temp_start * (temp_end / temp_start) ** ((episode-n_episodes/2) / (n_episodes/2))
        else:
            soft_q_agent.temperature = temp_start

        # Run episode
        while not gridworld.is_terminated and steps < max_steps_per_episode:
            s = gridworld.agent_pos
            a = soft_q_agent.sample_action(s)
            # Take action
            reward = gridworld.do_action(a)
            total_reward += reward
            steps += 1
            if episode % 10_000 == 0:
                print(gridworld)
                time.sleep(0.01)

        # Learn
        soft_q_agent.learn_from_episode()

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
            a = soft_q_agent.best_action(s)
            reward = gridworld.do_action(a)
            steps += 1
            clear_screen()
            print(gridworld)
            time.sleep(0.1)  # Slow down for visualization

        print(f"\nFinal Path Reward: {gridworld.total_reward:.2f}")
        print(f"Steps taken: {steps}")

    # Dopo il training, prima del plot:
    print("Path finale (greedy):")
    gridworld_test = GridWorld(CONFIG)
    gridworld_test.reset()
    steps = 0
    while not gridworld_test.is_terminated and steps < MAX_STEPS_PER_EPISODE:
        s = gridworld_test.agent_pos
        a = soft_q_agent.best_action(s)
        print(f"  step {steps}: pos={s}, action={a}, Q values: {[(tuple(la), soft_q_agent.Q(s, la)) for la in gridworld_test.get_legal_actions(s)]}")
        gridworld_test.do_action(a)
        steps += 1

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
    clear_screen()
    # main_QLEARNING()
    clear_screen()
    # main_VAPOR()
    main_SoftQLEARNING()