from __future__ import annotations
from typing import TYPE_CHECKING, Type
from plot_data import plot_population
import numpy as np
if TYPE_CHECKING:
    from simulation import Population


def powerset(length: int) -> set[set[int]]:
    powerset = set()
    for i in range(2 ** length):
        n = 0
        subset = []
        while i:
            if i % 2:
                subset.append(n)
            n += 1
            i //= 2
        powerset.add(tuple(subset))
    return powerset


def ess_search(population_type: Type[Population], outcome_matrix: np.ndarray):
    for behaviors in powerset(5):
        if len(behaviors) < 2:
            continue
        new_matrix = outcome_matrix[np.ix_(behaviors, behaviors)]
        try:
            result = np.linalg.solve(new_matrix, np.ones(len(behaviors)))
        except np.linalg.LinAlgError:
            continue
        result /= sum(result)
        if all(elem >= 0 for elem in result):
            population = population_type(size=100000, generation_count=500,
                                         fitness_offspring_factor=0.1, random_offspring_factor=0.001,
                                         outcome_matrix=outcome_matrix,
                                         behaviors=behaviors, starting_animal_ratios=tuple(result))
            population.run_simulation()
            plot_population(population)
