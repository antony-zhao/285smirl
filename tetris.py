import time

import pygame
import random
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete
import numpy as np

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

size = (400, 500)


class Figure:
    x = 0
    y = 0

    figures = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],
        [[4, 5, 9, 10], [2, 6, 5, 9]],
        [[6, 7, 9, 10], [1, 5, 6, 10]],
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        [[1, 2, 5, 6]],
    ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.figures) - 1)
        self.rotation = 0

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])


class Tetris:
    def __init__(self, height=20, width=10):
        self.level = 2
        self.score = 0
        self.field = []
        self.x = 100
        self.y = 60
        self.zoom = 20
        self.figure = Figure(3, 0)

        self.height = height
        self.width = width
        self.field = []
        self.score = 0
        self.done = False
        for i in range(height):
            new_line = []
            for j in range(width):
                new_line.append(0)
            self.field.append(new_line)

    def new_figure(self):
        self.figure = Figure(3, 0)

    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True
        return intersection

    def break_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        self.score += lines ** 2
        return lines

    def add_line(self, num_lines, ind):
        for _ in range(num_lines):
            del self.field[0]
            self.field.append([1 if i != ind else 0 for i in range(self.width)])

    def go_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        return self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            return self.freeze()
        return 0

    def freeze(self):
        lines = 0
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = 1
        lines += self.break_lines()
        self.new_figure()
        if self.intersects():
            self.done = True

        return lines

    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x

    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation


class TetrisEnv(ParallelEnv):
    # Currently designed for 2 agents but can be expanded to more
    def __init__(self, num_players=1, shape=(20, 10), full_obs=False):
        self.screen = None
        self._max_num_agents = num_players
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = self.possible_agents[:]
        self.action_space = Discrete(6)
        self.pad = (shape[0] - shape[1]) // 2
        shape = (shape[0], shape[0])
        self.observation_space = Box(0, 1, (1 if not full_obs else num_players,) + shape)
        self.action_spaces = {agent: self.action_space for agent in self.possible_agents}
        self.observation_spaces = {agent: self.observation_space for agent in self.possible_agents}

        self.garbage = {2: 1, 3: 2, 4: 4}
        self.targeting = {self.possible_agents[i]: self.agents[-i] for i in range(num_players)}
        self.max_timestep = 1000
        self.time = self.max_timestep
        self.full_obs = full_obs
        self.finished = {agent: False for agent in self.possible_agents}

        self.envs = {self.possible_agents[i]: Tetris(*shape) for i in range(num_players)}

    def step(self, actions):
        next_obs = {}
        reward = {}
        termination = {}
        truncation = {}
        info = {}
        for agent, action in actions.items():
            lines = 0
            reward[agent] = 1
            if not self.envs[agent].done:
                if action == 0:
                    pass
                elif action == 1:
                    self.envs[agent].go_side(1)
                elif action == 2:
                    self.envs[agent].go_side(-1)
                elif action == 3:
                    self.envs[agent].rotate()
                elif action == 4:
                    lines += self.envs[agent].go_down()
                elif action == 5:
                    lines += self.envs[agent].go_space()

                lines += self.envs[agent].go_down()

            garbage = self.garbage[lines] if lines >= 2 else 0
            if lines > 0:
                if any(1 not in line for line in self.envs[agent].field):
                    garbage += 4
                target = self.targeting[agent]
                if target != agent:
                    self.envs[target].add_lines(garbage)
                reward[agent] += garbage * 10
            if self.envs[agent].done:
                termination[agent] = True
                if not self.finished[agent]:
                    reward[agent] = -100
                    self.finished[agent] = True
                    self.handle_targeting()
            else:
                termination[agent] = False
            if self.time == 0:
                truncation[agent] = True
            else:
                truncation[agent] = False
            next_obs[agent] = self.get_obs(agent)
            info[agent] = None
        completed_envs = 0
        not_complete = 0
        for i, agent in enumerate(self.possible_agents):
            if self.envs[agent].done:
                completed_envs += 1
            not_complete = i
        if completed_envs == self.max_num_agents - 1 and self.max_num_agents > 1:
            reward[self.possible_agents[not_complete]] += 100
            termination[self.possible_agents[not_complete]] = True

        self.time -= 1

        return next_obs, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        obs = {}
        self.agents = self.possible_agents[:]
        self.targeting = {self.possible_agents[i]: self.agents[-i] for i in range(self.max_num_agents)}
        self.finished = {agent: False for agent in self.possible_agents}
        self.time = self.max_timestep
        for agent in self.possible_agents:
            self.envs[agent].__init__()

        for agent in self.possible_agents:
            obs[agent] = self.get_obs(agent)

        return obs

    def get_obs(self, agent):
        if not self.full_obs:
            obs = [np.pad(self.envs[agent].field, ((0, 0), (self.pad, self.pad)))]
        else:
            obs = [np.pad(self.envs[(i + self.possible_agents.index(agent)) % self.max_num_agents].field,
                          ((0, 0), (self.pad, self.pad))) for i in range(self.max_num_agents)]
        return np.asarray(obs)

    def handle_targeting(self):
        active_agents = []
        for agent in self.possible_agents:
            if not self.envs[agent].done:
                active_agents.append(agent)
        self.targeting = {active_agents[i]: active_agents[-i] for i in range(len(active_agents))}

    def render(self, agent=0):
        pygame.init()
        if self.screen is None:
            self.screen = pygame.display.set_mode(size)

        self.screen.fill(WHITE)

        game = self.envs[self.possible_agents[agent]]

        for i in range(game.height):
            for j in range(game.width):
                pygame.draw.rect(self.screen, GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom],
                                 1)
                if game.field[i][j] > 0:
                    pygame.draw.rect(self.screen, BLACK,
                                     [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2,
                                      game.zoom - 1])

        if game.figure is not None:
            for i in range(4):
                for j in range(4):
                    p = i * 4 + j
                    if p in game.figure.image():
                        pygame.draw.rect(self.screen, BLACK,
                                         [game.x + game.zoom * (j + game.figure.x) + 1,
                                          game.y + game.zoom * (i + game.figure.y) + 1,
                                          game.zoom - 2, game.zoom - 2])

        pygame.display.flip()
        time.sleep(0.1)

    def close(self):
        pygame.display.quit()
        self.screen = None


if __name__ == "__main__":
    pass
