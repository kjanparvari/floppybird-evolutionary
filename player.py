import random

import numpy
import pygame
import numpy as np

from nn import NeuralNetwork
from config import CONFIG


class Player:

    def __init__(self, mode, control=False):

        self.control = control  # if True, playing mode is activated. else, AI mode.
        self.pos = [100, 275]  # position of the agent
        self.direction = -1  # if 1, goes upwards. else, goes downwards.
        self.v = 0  # vertical velocity
        self.g = 9.8  # gravity constant
        self.mode = mode  # game mode

        # neural network architecture (AI mode)
        layer_sizes = self.init_network(mode)

        self.nn = NeuralNetwork(layer_sizes)
        self.fitness = 0  # fitness of agent

    def move(self, box_lists, camera, events=None):

        if len(box_lists) != 0:
            if box_lists[0].x - camera + 60 < self.pos[0]:
                box_lists.pop(0)

        mode = self.mode

        # manual control
        if self.control:
            self.get_keyboard_input(mode, events)

        # AI control
        else:
            agent_position = [camera + self.pos[0], self.pos[1]]
            self.direction = self.think(mode, box_lists, agent_position, self.v)

        # game physics
        if mode == 'gravity' or mode == 'helicopter':
            self.v -= self.g * self.direction * (1 / 60)
            if self.v > CONFIG['max_vel']:
                self.v = CONFIG['max_vel']
            elif self.v < -CONFIG['max_vel']:
                self.v = -CONFIG['max_vel']
            self.pos[1] += self.v

        elif mode == 'thrust':
            self.v -= 6 * self.direction
            if self.v > CONFIG['max_vel']:
                self.v = CONFIG['max_vel']
            elif self.v < -CONFIG['max_vel']:
                self.v = -CONFIG['max_vel']
            self.pos[1] += self.v * (1 / 40)

        # collision detection
        is_collided = self.collision_detection(mode, box_lists, camera)

        return is_collided

    # reset agent parameters
    def reset_values(self):
        self.pos = [100, 275]
        self.direction = -1
        self.v = 0

    def get_keyboard_input(self, mode, events=None):

        if events is None:
            events = pygame.event.get()

        if mode == 'helicopter':
            self.direction = -1
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.direction = 1

        elif mode == 'thrust':
            self.direction = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.direction = 1
            elif keys[pygame.K_DOWN]:
                self.direction = -1

        for event in events:
            if event.type == pygame.KEYDOWN:

                if mode == 'gravity' and event.key == pygame.K_SPACE:
                    self.direction *= -1

    @staticmethod
    def init_network(mode):

        # you can change the parameters below
        layer_sizes = None
        if mode == 'gravity':
            layer_sizes = [5, 20, 1]
        elif mode == 'helicopter':
            layer_sizes = [5, 20, 1]
        elif mode == 'thrust':
            layer_sizes = [5, 20, 1]
        return layer_sizes

    def think(self, mode, box_lists, agent_position, velocity):
        # mode example: 'helicopter'
        # box_lists: an array of `BoxList` objects
        # agent_position example: [600, 250]
        # velocity example: 7
        if len(box_lists) == 0:
            nx0, nx1, ny0, ny1 = 1, 1, 0, 0
        elif len(box_lists) == 1:
            nx0 = (box_lists[0].x - agent_position[0]) / CONFIG['WIDTH']
            ny0 = (box_lists[0].gap_mid - agent_position[1]) / CONFIG['HEIGHT']
            nx1, ny1 = 1, 0
        else:
            nx0 = (box_lists[0].x - agent_position[0]) / CONFIG['WIDTH']
            ny0 = (box_lists[0].gap_mid - agent_position[1]) / CONFIG['HEIGHT']
            nx1 = (box_lists[1].x - agent_position[0]) / CONFIG['WIDTH']
            ny1 = (box_lists[1].gap_mid - agent_position[1]) / CONFIG['HEIGHT']

        input_list = numpy.array([
            # normalized velocity
            [velocity / CONFIG['max_vel']],
            # normalized boxes positions
            [nx0],
            [ny0],
            [nx1],
            [ny1]
        ])
        # print(velocity)
        nn_output = self.nn.forward(input_list)[0][0]
        # print(nn_output)
        if mode == 'helicopter' or mode == 'gravity':
            return 1 if nn_output >= 0.5 else -1
        else:
            return 1 if nn_output >= 0.667 else -1 if nn_output <= 0.333 else 0

    def collision_detection(self, mode, box_lists, camera):
        if mode == 'helicopter':
            rect = pygame.Rect(self.pos[0], self.pos[1], 100, 50)
        elif mode == 'gravity':
            rect = pygame.Rect(self.pos[0], self.pos[1], 70, 70)
        elif mode == 'thrust':
            rect = pygame.Rect(self.pos[0], self.pos[1], 110, 70)
        else:
            rect = pygame.Rect(self.pos[0], self.pos[1], 50, 50)
        is_collided = False

        if self.pos[1] < -60 or self.pos[1] > CONFIG['HEIGHT']:
            is_collided = True

        if len(box_lists) != 0:
            box_list = box_lists[0]
            for box in box_list.boxes:
                box_rect = pygame.Rect(box[0] - camera, box[1], 60, 60)
                if box_rect.colliderect(rect):
                    is_collided = True

        return is_collided

    def copy(self):
        new_player = Player(self.mode, self.control)
        new_player.pos = self.pos.copy()
        new_player.direction = self.direction
        new_player.v = self.v
        new_player.g = self.g
        new_player.nn = self.nn.copy()
        new_player.fitness = self.fitness
        return new_player
