import numpy as np
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
    for beh_id in population.behs:
        ax0.plot(population.history[beh_id], label=beh_names[beh_id], color=beh_colors[beh_id])
    ax0.set_ylim([0, population.size])
    ax0.set_ylabel("amount")
    ax0.set_xlabel("generation")
    anchored_text = AnchoredText(f"Population size: {population.size}     "
                                 f"Generations: {population.generation}\n"
                                 f"Fitness offspring factor: {population.fitness_offspring_factor}  "
                                 f"Random offspring factor: {population.random_offspring_factor}", loc="upper center")
    anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

    ax0.add_artist(anchored_text)
    ax0.legend()

    ax1 = plt.subplot2grid(gridsize, (5, 0), rowspan=4, colspan=4)
    ax1.set_title("Distribution of behaviors in the starting population")
    ax1.pie([population.history[beh_id][0] for beh_id in population.behs],
            labels=[beh_names[i] for i in population.behs], autopct='%1.2f%%',
            shadow=True, startangle=90, colors=beh_colors[population.behs])
    ax1.axis("equal")

    ax2 = plt.subplot2grid(gridsize, (5, 8), rowspan=4, colspan=4)
    ax2.set_title("Distribution of behaviors in the final population")
    ax2.pie([population.history[beh_id][-1] for beh_id in population.behs],
            labels=[beh_names[i] for i in population.behs], autopct='%1.2f%%',
            shadow=True, startangle=90, colors=beh_colors[population.behs])
    ax2.axis("equal")
    plt.show()


class Population:
    def __init__(self, size, fitness_offspring_factor, random_offspring_factor, behaviors=(0, 2),
                 starting_animal_ratios=(1, 1), gens_in_sim=500):
        self.generation = 0
        self.gens_in_sim = gens_in_sim
        self.size = size
        self.behs = np.array(behaviors)
        self.fitness_offspring = int(size * fitness_offspring_factor)
        self.random_offspring = int(size * random_offspring_factor)
        self.fitness_offspring_factor = fitness_offspring_factor
        self.random_offspring_factor = random_offspring_factor
        # self.overpopulation_factor = 0.05
        self.animals = np.zeros(max(self.behs) + 1)
        for i, ratio in enumerate(starting_animal_ratios):
            self.animals[behaviors[i]] = int(self.size * ratio / sum(starting_animal_ratios))
        while sum(self.animals) != self.size:
            self.animals[np.random.choice(self.behs)] += 1
        self.history = [[] for _ in range(max(behaviors) + 1)]
        self.last_gen_points = np.array([0] * (max(behaviors) + 1))
        self.all_points = 0
        self.update_history()

    def new_generation(self):
        for beh in self.behs:
            self.animals[beh] *= (1 - (self.fitness_offspring_factor + self.random_offspring_factor))
            self.animals[beh] += (self.last_gen_points[beh] / self.all_points) * self.fitness_offspring
            self.animals[beh] += self.random_offspring / len(self.behs)
            # self.animals[beh] *= (1 - self.overpopulation_factor)
            # self.animals[beh] += self.size * self.overpopulation_factor / len(self.behs)

        self.last_gen_points *= 0
        self.all_points = 0

        self.generation += 1
        self.update_history()

    def run_generation(self):
        for beh in self.behs:
            avg_result = 0
            for opponent in self.behs:
                avg_result += outcome_matrix[beh, opponent] * self.animals[opponent]
            avg_result /= self.size
            self.last_gen_points[beh] = int(avg_result * self.animals[beh])
            self.all_points += int(avg_result * self.animals[beh])

    def update_history(self):
        for beh in self.behs:
            self.history[beh].append(self.animals[beh])

    def run_simulation(self):
        for _ in tqdm(range(self.gens_in_sim)):
            population.run_generation()
            population.new_generation()

    def __str__(self):
        string = f"Population size: {self.size}\nGeneration: {self.generation}\n"
        string += f"Animal counts: {({beh_id: self.history[beh_id][-1] for beh_id in self.behs})}\n"
        string += f"Animal ratios: {({beh_id: self.history[beh_id][-1] / self.size for beh_id in self.behs})}\n"
        return string


"""
      ║ dove │ hawk │ ret. │ bully│prober│
══════╬══════╪══════╪══════╪══════╪══════╪       ╪══════╪       ╪═════╪ 
 dove ║  115 │  100 │  115 │  100 │  100 │       │  x1  │       │  n  │ 
──────╫──────┼──────┼──────┼──────┼──────┼       ┼──────┼       ┼─────┼ 
 hawk ║  150 │  75  │  75  │  150 │  75  │       │  x2  │       │  n  │ 
──────╫──────┼──────┼──────┼──────┼──────┼       ┼──────┼       ┼─────┼ 
 ret. ║  115 │  75  │  115 │  150 │  115 │   *   │  x3  │   =   │  n  │ 
──────╫──────┼──────┼──────┼──────┼──────┼       ┼──────┼       ┼─────┼ 
 bully║  150 │  100 │  100 │  115 │  100 │       │  x4  │       │  n  │ 
──────╫──────┼──────┼──────┼──────┼──────┼       ┼──────┼       ┼─────┼ 
prober║  150 │  75  │  115 │  150 │  115 │       │  x5  │       │  n  │
──────╫──────┼──────┼──────┼──────┼──────┼       ┼──────┼       ┼─────┼
"""
beh_names = ["doves", "hawks", "retaliators", "bullies", "probers"]
beh_colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
outcome_matrix = np.array([[115, 100, 115, 100, 100],
                           [150,  75,  75, 150,  75],
                           [115,  75, 115, 150, 115],
                           [150, 100, 100, 115, 100],
                           [150,  75, 115, 150, 115]])
population = Population(size=1000000, fitness_offspring_factor=0.1, random_offspring_factor=0.00,
                        behaviors=(0, 1, 2, 3, 4), starting_animal_ratios=(1, 1, 1, 1, 1), gens_in_sim=10000)

print(population)
population.run_simulation()
print(population)

plot_data(population)

