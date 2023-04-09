import numpy as np
from plot_data import plot_data
from abc import ABC, abstractmethod


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
    behavior_points: np.array

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

        self.history = [[] for _ in range(max(self.behaviors) + 1)]
        self.behavior_points = np.zeros(max(self.behaviors) + 1)

    @abstractmethod
    def new_generation(self) -> None: ...

    @abstractmethod
    def run_generation(self) -> None: ...

    def run_simulation(self):
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
        self.animals += (self.behavior_points / np.sum(self.behavior_points)) * self.fitness_offspring
        self.animals += self.random_offspring / len(self.animals)

        self.generation += 1
        self.update_history()

    def run_generation(self):
        avg_results = (self.outcome_matrix * self.animals).sum(axis=1) / self.size
        self.behavior_points = avg_results * self.animals

    def update_history(self):
        for i, beh in enumerate(self.behaviors):
            self.history[beh].append(self.animals[i])


class PopulationRand(Population):
    def __init__(self, size, generation_count, fitness_offspring_factor, random_offspring_factor, outcome_matrix,
                 behaviors, starting_animal_ratios):
        super().__init__(size, generation_count, fitness_offspring_factor, random_offspring_factor, outcome_matrix,
                         behaviors)
        self.fitness_offspring = int(self.fitness_offspring)
        self.random_offspring = int(self.random_offspring)
        self.animals = [np.random.choice(self.behaviors, p=np.array(starting_animal_ratios) / sum(starting_animal_ratios))
                        for _ in range(self.size)]
        self.update_history()

    def new_generation(self):

        # REMOVING ANIMALS

        for _ in range(self.random_offspring + self.fitness_offspring):
            self.animals.pop(np.random.randint(len(self.animals) - 1))

        # ADDING ANIMALS
        for random_animal_type in np.random.choice(self.behaviors, self.fitness_offspring,
                                                   p=self.behavior_points / sum(self.behavior_points)):
            self.animals.append(random_animal_type)

        for _ in range(self.random_offspring):
            self.animals.append(np.random.choice(self.behaviors))

        self.behavior_points *= 0

        np.random.shuffle(self.animals)
        self.generation += 1
        self.update_history()

    def run_generation(self):
        for i in range(self.size // 2):
            animal1 = self.animals[i * 2]
            animal2 = self.animals[i * 2 + 1]
            self.behavior_points[animal1] += self.outcome_matrix[animal1, animal2]
            self.behavior_points[animal2] += self.outcome_matrix[animal2, animal1]

    def update_history(self):
        for beh in self.behaviors:
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
    population = PopulationAnalytical(size=100000, generation_count=5000,
                                      fitness_offspring_factor=0.1, random_offspring_factor=0.001,
                                      outcome_matrix=outcome_matrix_complex,
                                      behaviors=(0, 2, 4), starting_animal_ratios=(1, 1, 1))

    population.run_simulation()
    plot_data(population)
    # ess_search(PopulationMath, outcome_matrix_complex)

"""
    (0, 1, 4)
    [[129.  119.5 117.2]
     [180.   80.5  81.1]
     [156.7  79.9 121.9]]
     
    [[0.42535135]
     [0.32693185]
     [0.2477168 ]]
     
    [[122.97108916]
     [122.97108916]
     [122.97108916]]
 """