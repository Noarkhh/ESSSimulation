import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm

matplotlib.use('TkAgg')


def plot_data(population):
    gridsize = (10, 14)
    fig = plt.figure(figsize=(10, 8))

    ax0 = plt.subplot2grid(gridsize, (0, 0), rowspan=4, colspan=14)
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
    ax0.legend(loc="upper right")

    ax1 = plt.subplot2grid(gridsize, (6, 0), rowspan=4, colspan=6)
    ax1.set_title("Distribution of behaviors in the starting population")
    rects1 = ax1.bar([beh_names[i] for i in population.behs],
                     [population.history[beh_id][0] / population.size for beh_id in population.behs],
                     color=beh_colors[population.behs])
    ax1.bar_label(rects1,
                  labels=[str(round(population.history[beh_id][0] / population.size * 100, 2)) + '%' for beh_id in
                          population.behs], fmt='%1.2f%%', padding=3)
    ax1.set_ylim([0, 1.1])
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))

    ax2 = plt.subplot2grid(gridsize, (6, 8), rowspan=4, colspan=6)
    ax2.set_title("Distribution of behaviors in the final population")
    rects2 = ax2.bar([beh_names[i] for i in population.behs],
                     [population.history[beh_id][-1] / population.size for beh_id in population.behs],
                     color=beh_colors[population.behs])
    ax2.bar_label(rects2,
                  labels=[str(round(population.history[beh_id][-1] / population.size * 100, 2)) + '%' for beh_id in
                          population.behs], padding=3)
    ax2.set_ylim([0, 1.1])
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
    plt.show()


class Population:
    def __init__(self, size, gens_in_sim, fitness_offspring_factor, random_offspring_factor, outcome_matrix,
                 behaviors, starting_animal_ratios):
        self.generation = 0
        self.gens_in_sim = gens_in_sim
        self.size = size
        self.behs = np.array(behaviors)
        self.outcome_matrix = outcome_matrix
        self.fitness_offspring = int(size * fitness_offspring_factor)
        self.random_offspring = int(size * random_offspring_factor)
        self.fitness_offspring_factor = fitness_offspring_factor
        self.random_offspring_factor = random_offspring_factor
        # self.overpopulation_factor = 0.05

        self.history = [[] for _ in range(max(self.behs) + 1)]
        self.last_gen_points = np.array([0] * (max(self.behs) + 1))
        self.all_points = 0

    def run_simulation(self):
        for _ in tqdm(range(self.gens_in_sim)):
            population.run_generation()
            population.new_generation()

    def __str__(self):
        string = f"Population size: {self.size}\nGeneration: {self.generation}\n"
        string += f"Animal counts: {({beh_id: self.history[beh_id][-1] for beh_id in self.behs})}\n"
        string += f"Animal ratios: {({beh_id: self.history[beh_id][-1] / self.size for beh_id in self.behs})}\n"
        return string


class PopulationMath(Population):
    def __init__(self, size, gens_in_sim, fitness_offspring_factor, random_offspring_factor, outcome_matrix,
                 behaviors, starting_animal_ratios):
        super().__init__(size, gens_in_sim, fitness_offspring_factor, random_offspring_factor, outcome_matrix,
                         behaviors, starting_animal_ratios)

        self.animals = np.zeros(max(self.behs) + 1)
        for i, ratio in enumerate(starting_animal_ratios):
            self.animals[behaviors[i]] = self.size * ratio / sum(starting_animal_ratios)
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
                avg_result += self.outcome_matrix[beh, opponent] * self.animals[opponent]
            avg_result /= self.size
            self.last_gen_points[beh] = avg_result * self.animals[beh]
            self.all_points += avg_result * self.animals[beh]

    def update_history(self):
        for beh in self.behs:
            self.history[beh].append(self.animals[beh])


class PopulationRand(Population):
    def __init__(self, size, gens_in_sim, fitness_offspring_factor, random_offspring_factor, outcome_matrix,
                 behaviors, starting_animal_ratios):
        super().__init__(size, gens_in_sim, fitness_offspring_factor, random_offspring_factor, outcome_matrix,
                         behaviors, starting_animal_ratios)

        self.animals = [np.random.choice(self.behs, p=np.array(starting_animal_ratios) / sum(starting_animal_ratios))
                        for _ in range(self.size)]
        self.update_history()

    def new_generation(self):

        # REMOVING ANIMALS

        for _ in range(self.random_offspring + self.fitness_offspring):
            self.animals.pop(np.random.randint(len(self.animals) - 1))

        # ADDING ANIMALS
        for random_animal_type in np.random.choice(self.behs, self.fitness_offspring,
                                                   p=self.last_gen_points / sum(self.last_gen_points)):
            self.animals.append(random_animal_type)

        for _ in range(self.random_offspring):
            self.animals.append(np.random.choice(self.behs))

        self.last_gen_points *= 0

        np.random.shuffle(self.animals)
        self.generation += 1
        self.update_history()

    def run_generation(self):
        for i in range(self.size // 2):
            animal1 = self.animals[i * 2]
            animal2 = self.animals[i * 2 + 1]
            self.last_gen_points[animal1] += self.outcome_matrix[animal1, animal2]
            self.last_gen_points[animal2] += self.outcome_matrix[animal2, animal1]

    def update_history(self):
        for beh in self.behs:
            self.history[beh].append(0)
        for animal in self.animals:
            self.history[animal][-1] += 1


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
outcome_matrix_simplified = np.array([[115, 100, 115, 100, 100],
                                      [150, 75, 75, 150, 75],
                                      [115, 75, 115, 150, 115],
                                      [150, 100, 100, 115, 100],
                                      [150, 75, 115, 150, 115]])

outcome_matrix_complex = np.array([[129, 119.5, 129, 119.5, 117.2],
                                   [180, 80.5, 81.9, 174.6, 81.10],
                                   [129, 77.7, 129, 157.1, 123.1],
                                   [180, 104.9, 111.9, 141.5, 111.2],
                                   [156.7, 79.9, 126.9, 159.4, 121.9]])
if __name__ == "__main__":
    population = PopulationMath(size=100000, gens_in_sim=1000,
                                fitness_offspring_factor=0.1, random_offspring_factor=0.001,
                                outcome_matrix=outcome_matrix_complex,
                                behaviors=(0, 1, 2, 3, 4), starting_animal_ratios=(1, 0, 0, 0, 0))
    population.run_simulation()
    plot_data(population)
