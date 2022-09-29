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
                 starting_animal_ratios=(1, 1), gens_in_sim=500):
        self.generation = 0
        self.gens_in_sim = gens_in_sim
        self.size = size
        self.behs = number_of_behaviors
        self.fitness_change = int(size * fitness_change_factor)
        self.random_change = int(size * random_change_factor)
        self.fitness_change_factor = fitness_change_factor
        self.random_change_factor = random_change_factor
        self.animals = np.array([int(self.size * ratio / sum(starting_animal_ratios)) for i, ratio in enumerate(starting_animal_ratios)])
        while sum(self.animals) != self.size:
            self.animals[np.random.randint(self.behs - 1)] += 1
        self.history = [[] for _ in range(number_of_behaviors)]
        self.last_gen_points = np.array([0] * number_of_behaviors)
        self.all_points = 0
        self.update_history()

    def new_generation(self):
        for beh in range(self.behs):
            self.animals[beh] *= (1 - (self.fitness_change_factor + self.random_change_factor))
            self.animals[beh] += (self.last_gen_points[beh] / self.all_points) * self.fitness_change
            self.animals[beh] += self.random_change / 3

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


beh_names = ["dove", "hawk", "retaliator"]
outcome_matrix = np.array([115, 100, 115,
                           150, 75,  75,
                           115, 75,  115]).reshape((3, 3))

population = Population(size=1000000, fitness_change_factor=0.1, random_change_factor=0.001,
                        number_of_behaviors=3, starting_animal_ratios=(3, 2, 1), gens_in_sim=2000)

print(population)
population.run_simulation()
print(population)

plot_data(population)

