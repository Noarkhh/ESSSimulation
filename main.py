import numpy as np
from random import shuffle, randrange, choices, choice
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm
matplotlib.use('TkAgg')


class Animal:
    def __init__(self):
        self.points = 0


class Hawk(Animal):
    def __init__(self):
        super().__init__()


class Dove(Animal):
    def __init__(self):
        super().__init__()


class Retaliator(Animal):
    def __init__(self):
        super().__init__()


class Population:
    def __init__(self, size, fitness_change_factor, random_change_factor, animal_types=(Dove, Hawk),
                 starting_animal_ratios=(1, 1), gens_in_sim=500):
        self.generation = 0
        self.gens_in_sim = gens_in_sim
        self.size = size
        self.fitness_change = int(size * fitness_change_factor)
        self.random_change = int(size * random_change_factor)
        self.animal_types = animal_types
        self.animals = [choices(self.animal_types, starting_animal_ratios, k=1)[0]() for _ in range(self.size)]
        self.history = {animal_type: [] for animal_type in self.animal_types}
        self.update_history()

    def new_generation(self):

        # REMOVING ANIMALS

        for _ in range(self.random_change + self.fitness_change):
            self.animals.pop(randrange(len(self.animals)))

        # ADDING ANIMALS

        points_sum = {animal_type: 0 for animal_type in self.animal_types}
        for animal in self.animals:
            points_sum[type(animal)] += animal.points

        for random_animal_type in choices(self.animal_types, (points_sum.values()), k=self.fitness_change):
            self.animals.append(random_animal_type())

        for _ in range(self.random_change):
            self.animals.append(choice(self.animal_types)())

        for animal in self.animals:
            animal.points = 0

        shuffle(self.animals)
        self.generation += 1
        self.update_history()

    def run_generation(self):
        for i in range(self.size // 2):
            animal1 = self.animals[i * 2]
            animal2 = self.animals[i * 2 + 1]
            animal1.points += outcome_matrix[type(animal1)][type(animal2)]
            animal2.points += outcome_matrix[type(animal2)][type(animal1)]

    def update_history(self):
        for animal_type in self.history.keys():
            self.history[animal_type].append(0)
        for animal in self.animals:
            self.history[type(animal)][-1] += 1

    def run_simulation(self):
        for _ in tqdm(range(self.gens_in_sim)):
            population.run_generation()
            population.new_generation()

    def __str__(self):
        string = f"Population size: {self.size}\nGeneration: {self.generation}\n"
        string += f"Animal counts: {({type_to_str[animal]: amount[-1] for animal, amount in self.history.items()})}\n"
        string += f"Animal ratios: {({type_to_str[animal]: amount[-1] / self.size for animal, amount in self.history.items()})}\n"
        return string


type_to_str = {Dove: "Dove", Hawk: "Hawk", Retaliator: "Retaliator"}

# outcome_matrix = {Hawk: {Hawk: (75, 75), Dove: (150, 100), Retaliator: (75, 75)},
#                   Dove: {Hawk: (100, 150), Dove: (115, 115), Retaliator: (115, 115)},
#                   Retaliator: {Hawk: (75, 75), Dove: (115, 115), Retaliator: (115, 115)}}
outcome_matrix = {Hawk:       {Hawk: 75,  Dove: 150, Retaliator: 75},
                  Dove:       {Hawk: 100, Dove: 115, Retaliator: 115},
                  Retaliator: {Hawk: 75,  Dove: 115, Retaliator: 115}}
# outcome_matrix = {Hawk:       {Hawk: 75,  Dove: 150},
#                   Dove:       {Hawk: 100, Dove: 115}}
# v = 50
# c = 100
# outcome_matrix = {Hawk:       {Hawk: 100 + (v - c) // 2, Dove: 100 + v,      Retaliator: 100 + (v - c) // 2},
#                   Dove:       {Hawk: 100,                Dove: 100 + v // 2, Retaliator: 100 + v // 2},
#                   Retaliator: {Hawk: 100 + (v - c) // 2, Dove: 100 + v // 2, Retaliator: 100 + v // 2}}

population = Population(size=50000, fitness_change_factor=0.1, random_change_factor=0.0001,
                        animal_types=(Dove, Hawk, Retaliator), starting_animal_ratios=(1, 2, 3), gens_in_sim=3000)

print(population)
population.run_simulation()
print(population)

gridsize = (9, 12)
fig = plt.figure(figsize=(10, 8))

ax0 = plt.subplot2grid(gridsize, (0, 0), rowspan=4, colspan=12)
ax0.set_title("Behaviors in the population over time")
for animal, record in population.history.items():
    ax0.plot(record, label=type_to_str[animal] + 's')
ax0.set_ylim([0, population.size])
ax0.set_ylabel("amount")
ax0.set_xlabel("generation")
anchored_text = AnchoredText(f"Population size: {population.size}\nGenerations: {population.generation}", loc="upper center")
anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

ax0.add_artist(anchored_text)
ax0.legend()

ax1 = plt.subplot2grid(gridsize, (5, 0), rowspan=4, colspan=4)
ax1.set_title("Distribution of behaviors in the starting population")
ax1.pie([record[0] for record in population.history.values()],
        labels=[type_to_str[animal] + 's' for animal in population.animal_types], autopct='%1.2f%%',
        shadow=True, startangle=90)
ax1.axis("equal")

ax2 = plt.subplot2grid(gridsize, (5, 8), rowspan=4, colspan=4)
ax2.set_title("Distribution of behaviors in the final population")
ax2.pie([record[-1] for record in population.history.values()],
        labels=[type_to_str[animal] + 's' for animal in population.animal_types], autopct='%1.2f%%',
        shadow=True, startangle=90)
ax2.axis("equal")

plt.show()


