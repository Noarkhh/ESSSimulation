import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_data import plot_data
from abc import ABC, abstractmethod

behavior_names = ["doves", "hawks", "retaliators", "bullies", "probers"]


class Population(ABC):
    generation: int
    generation_count: int
    size: int
    behaviors: np.array
    outcome_matrix: np.array
    fitness_offspring_factor: float
    random_offspring_factor: float
    fitness_offspring: float
    random_offspring: float
    history: list[list[float]]
    points: np.array

    def __init__(self,
                 size: int,
                 generation_count: int,
                 fitness_offspring_factor: float,
                 random_offspring_factor: float,
                 outcome_matrix: np.array,
                 behaviors: tuple) -> None:

        self.generation = 0
        self.generation_count = generation_count
        self.size = size
        self.behaviors = np.array(behaviors)
        self.outcome_matrix = outcome_matrix[np.ix_(np.array(behaviors), np.array(behaviors))]
        self.fitness_offspring_factor = fitness_offspring_factor
        self.random_offspring_factor = random_offspring_factor
        self.fitness_offspring = size * fitness_offspring_factor
        self.random_offspring = size * random_offspring_factor

        self.history = [[] for _ in self.behaviors]

    @abstractmethod
    def new_generation(self) -> None: ...

    @abstractmethod
    def run_generation(self) -> None: ...

    def history_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({behavior_names[self.behaviors[i]]: records for i, records in enumerate(self.history)})

    def run_simulation(self):
        self.history = [[] for _ in self.behaviors]
        for _ in range(self.generation_count):
            self.run_generation()
            self.new_generation()

    def __str__(self):
        string = f"Population size: {self.size}\nGeneration: {self.generation}\n"
        string += f"Animal counts: {({beh_id: self.history[beh_id][-1] for beh_id in self.behaviors})}\n"
        string += f"Animal ratios: {({beh_id: self.history[beh_id][-1] / self.size for beh_id in self.behaviors})}\n"
        return string


class PopulationAnalytical(Population):
    def __init__(self, size, generation_count, fitness_offspring_factor, random_offspring_factor, outcome_matrix,
                 behaviors, starting_animal_ratios):
        super().__init__(size, generation_count, fitness_offspring_factor, random_offspring_factor, outcome_matrix,
                         behaviors)
        self.animals = np.array(starting_animal_ratios) * self.size / np.sum(starting_animal_ratios)
        self.update_history()

    def new_generation(self):
        self.animals *= (1 - (self.fitness_offspring_factor + self.random_offspring_factor))
        self.animals += (self.points / np.sum(self.points)) * self.fitness_offspring
        self.animals += self.random_offspring / len(self.animals)

        self.generation += 1
        self.update_history()

    def run_generation(self):
        avg_results = (self.outcome_matrix * self.animals).sum(axis=1) / self.size
        self.points = avg_results * self.animals

    def update_history(self):
        for i, animal_count in enumerate(self.animals):
            self.history[i].append(animal_count)


class PopulationSimulated(Population):
    def __init__(self, size, generation_count, fitness_offspring_factor, random_offspring_factor, outcome_matrix,
                 behaviors, starting_animal_ratios):
        super().__init__(size, generation_count, fitness_offspring_factor, random_offspring_factor, outcome_matrix,
                         behaviors)
        self.fitness_offspring = int(self.fitness_offspring)
        self.random_offspring = int(self.random_offspring)
        self.behaviors_number = len(behaviors)
        self.animals = np.random.choice(self.behaviors_number, self.size,
                                        p=np.array(starting_animal_ratios) / sum(starting_animal_ratios))
        self.update_history()

    def new_generation(self):

        # REMOVING ANIMALS
        # for _ in range(self.random_offspring + self.fitness_offspring):
        #     self.animals.pop(np.random.randint(len(self.animals) - 1))
        #
        # # ADDING ANIMALS
        # for random_animal_type in np.random.choice(self.behaviors, self.fitness_offspring,
        #                                            p=self.points / sum(self.points)):
        #     self.animals.append(random_animal_type)
        #
        # for _ in range(self.random_offspring):
        #     self.animals.append(np.random.choice(self.behaviors))

        indices_to_replace = np.random.choice(self.size, size=self.random_offspring + self.fitness_offspring, replace=False)
        self.animals[indices_to_replace[:self.fitness_offspring]] = \
            np.random.choice(self.behaviors_number, self.fitness_offspring, p=self.points / sum(self.points))
        self.animals[indices_to_replace[self.fitness_offspring:]] = \
            np.random.randint(self.behaviors_number, size=self.random_offspring)

        # np.random.shuffle(self.animals)
        self.generation += 1
        self.update_history()

    def run_generation(self):
        self.points = np.zeros(self.behaviors_number)
        for i in range(self.size // 2):
            animal1 = self.animals[i * 2]
            animal2 = self.animals[i * 2 + 1]
            self.points[animal1] += self.outcome_matrix[animal1, animal2]
            self.points[animal2] += self.outcome_matrix[animal2, animal1]

    def update_history(self):
        for records in self.history:
            records.append(0)
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
    population = PopulationAnalytical(size=5000, generation_count=5000,
                                      fitness_offspring_factor=0.1, random_offspring_factor=0.00,
                                      outcome_matrix=outcome_matrix_complex,
                                      behaviors=(0, 2, 4), starting_animal_ratios=(1, 1, 1))

    population.run_simulation()
    plot_data(population)

    # ess_search(PopulationMath, outcome_matrix_complex)
