import random
import statistics

import numpy.random

from player import Player
import numpy as np
from config import CONFIG
import csv


class Evolution:

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    @staticmethod
    def calculate_fitness(players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def crossover(self, parent1, parent2):
        child = Player(self.mode)
        for layer_number in parent1.nn.weights.keys():
            if layer_number % 2 == 0:
                child.nn.weights[layer_number] = parent1.nn.weights[layer_number].copy()
                child.nn.biases[layer_number] = parent1.nn.biases[layer_number].copy()
            else:
                child.nn.weights[layer_number] = parent2.nn.weights[layer_number].copy()
                child.nn.biases[layer_number] = parent2.nn.biases[layer_number].copy()
        return child

    @staticmethod
    def mutate2(child: Player):
        mutation_probability = 0.8
        noise_range = .3
        if random.random() > mutation_probability:
            for layer_number in child.nn.weights.keys():
                child.nn.weights[layer_number] += np.random.normal(0, noise_range, child.nn.weights[layer_number].shape)
                child.nn.biases[layer_number] += np.random.normal(0, noise_range, child.nn.biases[layer_number].shape)
        return child

    @staticmethod
    def mutate(child: Player):
        def generate_noise(shape):
            mutation_probability = 0.8
            noise_range = .3
            noise = np.random.normal(0, noise_range, shape)
            noise = np.where(np.random.random(noise.shape) < (1 - mutation_probability), 0, noise)
            return noise

        for layer_number in child.nn.weights.keys():
            child.nn.weights[layer_number] += generate_noise(child.nn.weights[layer_number].shape)
            child.nn.biases[layer_number] += generate_noise(child.nn.biases[layer_number].shape)
        return child

    def generate_new_population(self, num_players, prev_players=None):
        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]
        else:
            new_players = []
            population_fitness = sum([player.fitness for player in prev_players])
            player_probabilities = [player.fitness / population_fitness for player in prev_players]
            parents = []
            qt_parameter = 5
            p = 0.5
            while len(parents) != 2 * num_players:
                candidate = np.random.choice(prev_players, qt_parameter, replace=False).tolist()
                candidate.sort(key=lambda x: x.fitness, reverse=True)
                for i in range(len(candidate)):
                    if random.random() < p * (1 - p) ** i:
                        parents.append(candidate[i])
                        break
            # parents = np.random.choice(prev_players, 2*num_players, p=player_probabilities, replace=True)
            crossover_probability = 0.4
            for i in range(num_players):
                if random.random() < crossover_probability:
                    new_players.append(self.crossover(parents[i * 2], parents[i * 2 + 1]))
                else:
                    new_players.append(parents[i * 2].copy())
            for i in range(num_players):
                self.mutate2(new_players[i])
            print(num_players)
            print(len(new_players))
            return new_players

    @staticmethod
    def next_population_selection(players, num_players):
        def get_fitness(player: Player):
            return player.fitness

        players.sort(key=get_fitness, reverse=True)
        _mean = statistics.mean([player.fitness for player in players])
        with open('records.csv', 'a+', newline='') as _file:
            writer = csv.writer(_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([players[0].fitness, players[len(players) - 1].fitness, _mean])
        return players[: num_players]
