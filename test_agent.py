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

import pygame

BASE_SIZE = 6
N_EPISODES = 50
MAX_STEPS_PER_EPISODE = 2 * (BASE_SIZE + 1)

CONFIG = GridWorldConfig(
    size=BASE_SIZE, 
    p_walls=0.30, 
    agent_start=np.array((0, 0)),
    step_penalty=-(2**-10), 
    small_treasure_rew=1e-3, 
    treasure_rew=1,
    sd_small_treasure=1e-3, 
    sd_treasure=1e-3, 
    temperature=0.0,
    gamma=0.995, 
    random_state=1
)

def gui(gridworld, agent, size_gui = 320):
    print("\n=== Final Learned Path ===")
    
    # Reset environment for final demonstration
    gridworld = GridWorld(CONFIG)
    gridworld.reset()
    
    size_image = size_gui / (gridworld.size + 2)

    pygame.init()
    pygame.display.set_caption("=== Final Learned Path ===")
    screen = pygame.display.set_mode((size_gui, size_gui))
    clock = pygame.time.Clock()
    walls = []
    
    for i in range(gridworld.size):
        for j in range(gridworld.size):
            if gridworld.grid[i, j] == gridworld.WALL:
                walls.append(pygame.Rect((j+1)*size_image, (i+1)*size_image, size_image, size_image))
    
    for i in range(gridworld.size + 2):
        walls.append(pygame.Rect(i*size_image, 0, size_image + 2, size_image))
        walls.append(pygame.Rect(i*size_image, size_gui-size_image, size_image + 2, size_image))
    
    for i in range(gridworld.size):
        walls.append(pygame.Rect(0, (i+1)*size_image, size_image, size_image + 2))
        walls.append(pygame.Rect(size_gui-size_image, (i+1)*size_image, size_image, size_image + 2))
            
    big_treasure = pygame.Rect(size_gui - 2*size_image, size_gui - 2*size_image, size_image, size_image)
    small_treasure = pygame.Rect(size_gui - 2*size_image, size_image, size_image, size_image)
    
    player = pygame.Rect(size_image, size_image, size_image, size_image)
    
    steps = 0
    while not gridworld.is_terminated and steps < MAX_STEPS_PER_EPISODE:
        s = gridworld.agent_pos
        if isinstance(agent, Vapor):
            a = agent.best_action(steps, s)
        else:
            a = agent.best_action(s)
        reward = gridworld.do_action(a)
        steps += 1
    
        player.x = (gridworld.agent_pos[1] + 1) * size_image
        player.y = (gridworld.agent_pos[0] + 1) * size_image
    
        time.sleep(1.0)  # Slow down for visualization
        #clear_screen()
        #print(gridworld)
        clock.tick(60)
    
        # Draw the scene
        screen.fill((0, 0, 0))
        for wall in walls:
            pygame.draw.rect(screen, (255, 255, 255), wall)
        pygame.draw.rect(screen, (0, 255, 0), big_treasure)
        pygame.draw.rect(screen, (0, 128, 0), small_treasure)
        pygame.draw.rect(screen, (255, 200, 0), player)
        pygame.display.flip()
        clock.tick(360)
    
    pygame.quit()
    print(f"\nFinal Path Reward: {gridworld.total_reward:.2f}")
    print(f"Steps taken: {steps}")

def clear_screen():
    command = "cls" if os.name == "nt" else "clear"
    subprocess.run([command], shell=True)


def main_QLEARNING():
    # Initialize the environment
    gridworld = GridWorld(CONFIG)

    # Terminal states are the treasure positions
    terminal_states = [gridworld.treasure_pos, gridworld.small_treasure_pos]

    # Initialize Q-learning agent
    q_agent = QLearning(gridworld, terminal_states, alpha=1e-3)
    n_episodes = 1000 * N_EPISODES
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
        #q_agent.alpha = 0.1 * (1 - episode / n_episodes)  # Decaying learning rate

        # Run episode
        while not gridworld.is_terminated and steps < max_steps_per_episode:
            s = gridworld.agent_pos
            a = q_agent.best_action_epsilon_greedy(s, epsilon=0.9)
            # Take action
            reward = gridworld.do_action(a)
            total_reward += reward
            steps += 1
            if episode % 1000 == 0:
                print(gridworld)
                time.sleep(0.01)

        if gridworld.agent_pos[1] != gridworld.size - 1:
            tmp = gridworld.current_episode[-1]
            tmp_2 = (tmp[0], tmp[1], tmp[2], -0.125)
            gridworld.current_episode.pop(-1)
            gridworld.current_episode.append(tmp_2)
        elif gridworld.agent_pos[0] != gridworld.size - 1 and gridworld.agent_pos[0] != 0:
            tmp = gridworld.current_episode[-1]
            tmp_2 = (tmp[0], tmp[1], tmp[2], -0.125)
            gridworld.current_episode.pop(-1)
            gridworld.current_episode.append(tmp_2)

        # Learn
        q_agent.learn_from_episode()

        # Store episode reward
        episode_rewards.append(total_reward)


    # After training
    if show_final_path:
        gui(gridworld, q_agent, size_gui=640)

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
    # print(gridworld)

    # Terminal states are the treasure positions
    terminal_states = [a2idx(gridworld.treasure_pos), a2idx(gridworld.small_treasure_pos)]
    n_episodes = N_EPISODES
    show_final_path = True
    episode_rewards = []

    # Initialize VAPOR agent
    VAPOR_agent = Vapor(gridworld, terminal_states, horizon=MAX_STEPS_PER_EPISODE)

    for ep in range(n_episodes):
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
            # if ep % 10 == 0:
            #     print(gridworld)
            #     time.sleep(0.1)

        if gridworld.agent_pos[1] != gridworld.size - 1:
            tmp = gridworld.current_episode[-1]
            tmp_2 = (tmp[0], tmp[1], tmp[2], -0.125)
            gridworld.current_episode.pop(-1)
            gridworld.current_episode.append(tmp_2)
        elif gridworld.agent_pos[0] != gridworld.size - 1 and gridworld.agent_pos[0] != 0:
            tmp = gridworld.current_episode[-1]
            tmp_2 = (tmp[0], tmp[1], tmp[2], -0.125)
            gridworld.current_episode.pop(-1)
            gridworld.current_episode.append(tmp_2)

        # Learn and store episode reward
        VAPOR_agent.learn_from_episode()
        print(f"Episode={ep} Reward={total_reward} Steps={steps}")
        episode_rewards.append(total_reward)

    input("Press enter to continue...")

    # After training
    if show_final_path:
        # clear_screen()
        gui(gridworld, VAPOR_agent, size_gui=640)

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

    temp_start = 1e-1
    temp_end = 1e-4

    # Initialize Q-learning agent
    soft_q_agent = SoftQLearning(gridworld, terminal_states, alpha=1, temperature=temp_start)
    n_episodes = 10_000
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
        soft_q_agent.temperature = temp_start * (temp_end / temp_start) ** (episode/n_episodes)


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

        if gridworld.agent_pos[1] != gridworld.size - 1:
            tmp = gridworld.current_episode[-1]
            tmp_2 = (tmp[0], tmp[1], tmp[2], -0.125)
            gridworld.current_episode.pop(-1)
            gridworld.current_episode.append(tmp_2)
        elif gridworld.agent_pos[0] != gridworld.size - 1 and gridworld.agent_pos[0] != 0:
            tmp = gridworld.current_episode[-1]
            tmp_2 = (tmp[0], tmp[1], tmp[2], -0.125)
            gridworld.current_episode.pop(-1)
            gridworld.current_episode.append(tmp_2)

        # Learn
        soft_q_agent.learn_from_episode()

        # Store episode reward
        episode_rewards.append(total_reward)


    # After training
    if show_final_path:
        gui(gridworld, soft_q_agent, size_gui=640)

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
    # main_VAPOR()
    main_SoftQLEARNING()