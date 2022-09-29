import numpy as np
from random import shuffle, randrange, choices, choice
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm
matplotlib.use('TkAgg')


def plot_data(population):
    gridsize = (9, 12)
    fig = plt.figure(figsize=(10, 8))

    ax0 = plt.subplot2grid(gridsize, (0, 0), rowspan=4, colspan=12)
    ax0.set_title("Behaviors in the population over time")
    for beh_id, record in enumerate(population.history):
        ax0.plot(record, label=beh_names[beh_id] + 's')
    ax0.set_ylim([0, population.size])
    ax0.set_ylabel("amount")
    ax0.set_xlabel("generation")
    anchored_text = AnchoredText(f"Population size: {population.size}\nGenerations: {population.generation}", loc="upper center")
    anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

    ax0.add_artist(anchored_text)
    ax0.legend()

    ax1 = plt.subplot2grid(gridsize, (5, 0), rowspan=4, colspan=4)
    ax1.set_title("Distribution of behaviors in the starting population")
    ax1.pie([record[0] for record in population.history],
            labels=[beh_names[i] + 's' for i in range(population.behs)], autopct='%1.2f%%',
            shadow=True, startangle=90)
    ax1.axis("equal")

    ax2 = plt.subplot2grid(gridsize, (5, 8), rowspan=4, colspan=4)
    ax2.set_title("Distribution of behaviors in the final population")
    ax2.pie([record[-1] for record in population.history],
            labels=[beh_names[i] + 's' for i in range(population.behs)], autopct='%1.2f%%',
            shadow=True, startangle=90)
    ax2.axis("equal")
    plt.show()


class Population:
    def __init__(self, size, fitness_change_factor, random_change_factor, number_of_behaviors=2,
                 starting_animal_ratios=(1/2, 1/2), gens_in_sim=500):
        self.generation = 0
        self.gens_in_sim = gens_in_sim
        self.size = size
        self.behs = number_of_behaviors
        self.fitness_change = int(size * fitness_change_factor)
        self.random_change = int(size * random_change_factor)
        self.animals = np.array([int(self.size * ratio) for ratio in starting_animal_ratios])
        while sum(self.animals) != self.size:
            self.animals[np.random.randint(self.behs - 1)] += 1
        self.history = [[] for _ in range(number_of_behaviors)]
        self.last_gen_points = np.array([0] * number_of_behaviors)
        self.all_points = 0
        self.update_history()

    def new_generation(self):

        # REMOVING ANIMALS
        for _ in range(self.random_change + self.fitness_change):
            self.animals[choices([i for i in range(self.behs)], self.animals)[0]] -= 1

        # ADDING ANIMALS
        # print(self.last_gen_points, self.all_points)
        for random_animal_type in choices([i for i in range(self.behs)], self.last_gen_points, k=self.fitness_change):
            self.animals[random_animal_type] += 1

        for _ in range(self.random_change):
            self.animals[np.random.randint(self.behs - 1)] += 1

        self.last_gen_points *= 0
        self.all_points = 0

        self.generation += 1
        self.update_history()

    def run_generation(self):
        for beh in range(self.behs):
            avg_result = 0
            for opponent in range(self.behs):
                avg_result += outcome_matrix[beh, opponent] * self.animals[opponent]
            avg_result /= self.size
            self.last_gen_points[beh] = int(avg_result * self.animals[beh])
            self.all_points += int(avg_result * self.animals[beh])

        # for i in range(self.size // 2):
        #     animal1 = self.animals[i * 2]
        #     animal2 = self.animals[i * 2 + 1]
        #     self.last_gen_points[animal1] += outcome_matrix[animal1, animal2]
        #     self.all_points += outcome_matrix[animal1, animal2]
        #     self.last_gen_points[animal2] += outcome_matrix[animal2, animal1]
        #     self.all_points += outcome_matrix[animal2, animal1]

    def update_history(self):
        for beh in range(self.behs):
            self.history[beh].append(self.animals[beh])

    def run_simulation(self):
        for _ in tqdm(range(self.gens_in_sim)):
            population.run_generation()
            population.new_generation()

    def __str__(self):
        string = f"Population size: {self.size}\nGeneration: {self.generation}\n"
        string += f"Animal counts: {({beh_id: record[-1] for beh_id, record in enumerate(self.history)})}\n"
        string += f"Animal ratios: {({beh_id: record[-1] / self.size for beh_id, record in enumerate(self.history)})}\n"
        return string


beh_names = ["hawk", "dove", "retaliator"]
outcome_matrix = np.array([75,  150, 75,
                           100, 115, 115,
                           75,  115, 115]).reshape((3, 3))
# v = 50
# c = 100
# outcome_matrix = {Hawk:       {Hawk: 100 + (v - c) // 2, Dove: 100 + v,      Retaliator: 100 + (v - c) // 2},
#                   Dove:       {Hawk: 100,                Dove: 100 + v // 2, Retaliator: 100 + v // 2},
#                   Retaliator: {Hawk: 100 + (v - c) // 2, Dove: 100 + v // 2, Retaliator: 100 + v // 2}}

population = Population(size=10000, fitness_change_factor=0.1, random_change_factor=0.001,
                        number_of_behaviors=2, starting_animal_ratios=(3/6, 3/6,), gens_in_sim=1000)

print(population)
population.run_simulation()
print(population)

plot_data(population)

