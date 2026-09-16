import time
import os
import argparse
import subprocess
import matplotlib.pyplot as plt
import numpy as np

from utils import a2idx
from enviroment import GridWorld
from utils import GridWorldConfig
from agents import QLearning, Vapor, SoftQLearning
from tqdm import tqdm

import pygame

BASE_SIZE = 5
N_EPISODES = 500
MAX_STEPS_PER_EPISODE = 8

CONFIG = GridWorldConfig(
    size=BASE_SIZE,
    p_walls=0.30,
    agent_start=np.array((0, 0)),
    step_penalty=-(2**-8),
    small_treasure_rew=1e-3,
    treasure_rew=1,
    sd_small_treasure=1e-1,
    sd_treasure=1e-1,
    temperature=0.0,
    gamma=0.995,
    random_state=1,
)


def get_gui_direction(gridworld, action):

    if np.array_equal(action, gridworld.UP):
        return "RIGHT"

    if np.array_equal(action, gridworld.DOWN):
        return "LEFT"

    if np.array_equal(action, gridworld.LEFT):
        return "UP"

    if np.array_equal(action, gridworld.RIGHT):
        return "DOWN"

    return None


def value_to_color(value, vmin, vmax):

    if vmax < 1e-3:
        vmax = vmin

    if vmax <= vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)

    t = np.clip(t, 0.0, 1.0)

    # Blue -> cyan -> yellow
    if t < 0.5:
        u = t / 0.5
        r = int(30 + (50 - 30) * u)
        g = int(80 + (200 - 80) * u)
        b = int(180 + (255 - 180) * u)

    else:
        u = (t - 0.5) / 0.5
        r = int(50 + (255 - 50) * u)
        g = int(200 + (220 - 200) * u)
        b = int(255 - (255 - 40) * u)

    return (r, g, b)


def draw_value_in_triangle(
    screen, points, value, direction, font, selected=False, vmin=0.0, vmax=1.0
):
    top_left, top_right, bottom_left, bottom_right, center = points

    if direction == "UP":
        polygon = [top_left, top_right, center]
        text_pos = (center[0], top_left[1] + 0.30 * (center[1] - top_left[1]))

    elif direction == "DOWN":
        polygon = [bottom_left, bottom_right, center]
        text_pos = (center[0], center[1] + 0.70 * (bottom_left[1] - center[1]))

    elif direction == "LEFT":
        polygon = [top_left, bottom_left, center]
        text_pos = (top_left[0] + 0.30 * (center[0] - top_left[0]), center[1])

    elif direction == "RIGHT":
        polygon = [top_right, bottom_right, center]
        text_pos = (center[0] + 0.70 * (top_right[0] - center[0]), center[1])

    else:
        return

    color = value_to_color(value, vmin, vmax)

    # Selected action gets a black border
    border_color = (20, 20, 20)
    border_width = 2 if selected else 1
    pygame.draw.polygon(screen, color, polygon)
    pygame.draw.polygon(screen, border_color, polygon, border_width)
    text = font.render(f"{value:.3f}", True, (0, 0, 0))
    text_rect = text.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
    screen.blit(text, text_rect)


def get_action_value(agent, state, action, step=0):

    if isinstance(agent, Vapor):
        key = (step, a2idx(state), a2idx(action))
        if key not in agent.qstate_to_idx:
            return None

        return agent.lamb(step, state, action)

    elif isinstance(agent, (QLearning, SoftQLearning)):
        return agent.Q(state, action)

    raise TypeError(f"Unsupported agent type: {type(agent).__name__}")


def draw_grid_values(
    screen, gridworld, for_lambdas, agent, step, font, selected_action=None
):
    for state in for_lambdas.keys():
        state_np = np.array(state)
        cell = gridworld.grid[a2idx(state_np)]

        if cell == gridworld.WALL:
            continue

        if cell == gridworld.TREASURE:
            continue

        if cell == gridworld.SMALL_TREASURE:
            continue

        legal_actions = gridworld.get_legal_actions(state_np)

        # First collect values

        action_values = []

        for action in legal_actions:
            value = get_action_value(agent, state_np, action, step)
            if value is None:
                continue

            direction = get_gui_direction(gridworld, action)
            if direction is None:
                continue

            action_values.append((action, direction, value))

        if not action_values:
            continue

        values = [value for _, _, value in action_values]
        vmin = min(values)
        vmax = max(values)

        # Draw

        for action, direction, value in action_values:
            selected = (
                selected_action is not None
                and np.array_equal(state_np, gridworld.agent_pos)
                and np.array_equal(action, selected_action)
            )

            draw_value_in_triangle(
                screen=screen,
                points=for_lambdas[state],
                value=value,
                direction=direction,
                font=font,
                selected=selected,
                vmin=vmin,
                vmax=vmax,
            )


def draw_walls_and_border(screen, gridworld, size_image):

    border_width = int(size_image)

    # Top
    pygame.draw.rect(
        screen,
        (50, 50, 50),
        pygame.Rect(0, 0, (gridworld.size + 2) * size_image, size_image),
    )

    # Bottom
    pygame.draw.rect(
        screen,
        (50, 50, 50),
        pygame.Rect(
            0,
            (gridworld.size + 1) * size_image,
            (gridworld.size + 2) * size_image,
            size_image,
        ),
    )

    # Left
    pygame.draw.rect(
        screen,
        (50, 50, 50),
        pygame.Rect(0, 0, size_image, (gridworld.size + 2) * size_image),
    )

    # Right
    pygame.draw.rect(
        screen,
        (50, 50, 50),
        pygame.Rect(
            (gridworld.size + 1) * size_image,
            0,
            size_image,
            (gridworld.size + 2) * size_image,
        ),
    )

    for r in range(gridworld.size):
        for c in range(gridworld.size):

            if gridworld.grid[r, c] == gridworld.WALL:

                wall = pygame.Rect(
                    (c + 1) * size_image,
                    (r + 1) * size_image,
                    size_image + 1,
                    size_image + 1,
                )

                pygame.draw.rect(screen, (50, 50, 50), wall)


def draw_treasures(screen, gridworld, size_image):

    # Big treasure
    treasure = pygame.Rect(
        (gridworld.treasure_pos[1] + 1) * size_image,
        (gridworld.treasure_pos[0] + 1) * size_image,
        size_image + 1,
        size_image + 1,
    )

    pygame.draw.rect(screen, (0, 255, 0), treasure)

    # Small treasure
    small_treasure = pygame.Rect(
        (gridworld.small_treasure_pos[1] + 1) * size_image,
        (gridworld.small_treasure_pos[0] + 1) * size_image,
        size_image + 1,
        size_image + 1,
    )

    pygame.draw.rect(screen, (0, 128, 0), small_treasure)


def gui(gridworld, agent, size_gui=640):

    pygame.init()
    gridworld.reset()
    size_image = size_gui / (gridworld.size + 2)
    screen = pygame.display.set_mode((int(size_gui), int(size_gui)))
    pygame.display.set_caption("GridWorld")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", max(10, int(size_image * 0.1)))
    rect_surface = pygame.Surface((size_image, size_image), pygame.SRCALPHA)
    for_lambdas = {}

    for r in range(gridworld.size):
        for c in range(gridworld.size):

            if gridworld.grid[r, c] == gridworld.WALL:
                continue

            x = (c + 1) * size_image
            y = (r + 1) * size_image

            for_lambdas[(r, c)] = [
                # top-left
                (x, y),
                # top-right
                (x + size_image, y),
                # bottom-left
                (x, y + size_image),
                # bottom-right
                (x + size_image, y + size_image),
                # center
                (x + size_image / 2, y + size_image / 2),
            ]

    player = pygame.Rect(
        int(size_image), int(size_image), int(size_image + 1), int(size_image + 1)
    )

    steps = 0
    running = True
    not_terminal_state = True

    while running and steps < MAX_STEPS_PER_EPISODE and not_terminal_state:

        if steps == 0:
            time.sleep(10)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        state = gridworld.agent_pos.copy()
        if isinstance(agent, Vapor):
            action = agent.best_action(steps, state)

        else:
            action = agent.best_action(state)

        screen.fill((255, 255, 255))
        draw_walls_and_border(screen, gridworld, size_image)
        draw_grid_values(
            screen, gridworld, for_lambdas, agent, steps, font, selected_action=action
        )

        draw_treasures(screen, gridworld, size_image)

        # Player
        player.x = int((gridworld.agent_pos[1] + 1) * size_image)
        player.y = int((gridworld.agent_pos[0] + 1) * size_image)
        """
        pygame.draw.rect(
            screen,
            (255, 120, 255, 64),
            player
        )
        """
        pygame.draw.rect(
            rect_surface, (255, 120, 255, 128), (0, 0, size_image, size_image)
        )

        screen.blit(rect_surface, (player.x, player.y))
        pygame.display.flip()

        clock.tick(60)
        time.sleep(5.0)

        gridworld.do_action(action)

        steps += 1
        player.x = int((gridworld.agent_pos[1] + 1) * size_image)
        player.y = int((gridworld.agent_pos[0] + 1) * size_image)

        terminal = lambda x: np.array_equal(x, np.array(gridworld.agent_pos))
        for ts in [gridworld.small_treasure_pos, gridworld.treasure_pos]:
            if terminal(ts):
                print(terminal(ts))
                not_terminal_state = not not_terminal_state
                break

    pygame.quit()


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
        # q_agent.alpha = 0.1 * (1 - episode / n_episodes)  # Decaying learning rate

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
        elif (
            gridworld.agent_pos[0] != gridworld.size - 1 and gridworld.agent_pos[0] != 0
        ):
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
    # plt.savefig(f"./figures/run_q_{time.time_ns()}")


def main_VAPOR():
    # Initialize the environment
    gridworld = GridWorld(CONFIG)
    # print(gridworld)

    # Terminal states are the treasure positions
    terminal_states = [
        a2idx(gridworld.treasure_pos),
        a2idx(gridworld.small_treasure_pos),
    ]
    n_episodes = N_EPISODES
    show_final_path = True
    episode_rewards = []

    # Initialize VAPOR agent
    VAPOR_agent = Vapor(
        gridworld, terminal_states, horizon=MAX_STEPS_PER_EPISODE, repbuffer_size=2
    )

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
            reward = gridworld.do_action(a)  # Take action
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
        elif (
            gridworld.agent_pos[0] != gridworld.size - 1 and gridworld.agent_pos[0] != 0
        ):
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

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    # plt.savefig(f"./figures/run_vapor_{time.time_ns()}")


def main_SoftQLEARNING():
    # Initialize the environment
    gridworld = GridWorld(CONFIG)

    # Terminal states are the treasure positions
    terminal_states = [gridworld.treasure_pos, gridworld.small_treasure_pos]

    temp_start = 1e-1
    temp_end = 1e-4

    # Initialize Q-learning agent
    soft_q_agent = SoftQLearning(
        gridworld, terminal_states, alpha=1, temperature=temp_start
    )
    n_episodes = 50_000
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
        soft_q_agent.temperature = temp_start * (temp_end / temp_start) ** (
            episode / n_episodes
        )

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
        elif (
            gridworld.agent_pos[0] != gridworld.size - 1 and gridworld.agent_pos[0] != 0
        ):
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
        print(
            f"  step {steps}: pos={s}, action={a}, Q values: {[(tuple(la), soft_q_agent.Q(s, la)) for la in gridworld_test.get_legal_actions(s)]}"
        )
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
    
    parser = argparse.ArgumentParser(description="Run different agents")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--q", action="store_true", help="Run Q-Learning")
    group.add_argument("--vapor", action="store_true", help="Run VAPOR")
    group.add_argument("--soft", action="store_true", help="Run Soft Q-Learning")
    args = parser.parse_args()

    if args.q:
        main_QLEARNING()
    elif args.vapor:
        main_VAPOR()
    elif args.soft:
        main_SoftQLEARNING()
    else:
        # Default behavior if no flag is provided
        print("No flag provided. Defaulting to VAPORs")
        main_VAPOR()