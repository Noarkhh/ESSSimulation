from plot_data import plot_data
import numpy as np


def ess_search(population_type, outcome_matrix):
    behaviors = [0, 1, 2, 3, 4]
    powerset = []
    for i in range(2 ** 5):
        n = 0
        subset = []
        while i:
            if i % 2:
                subset.append(n)
            n += 1
            i //= 2
        powerset.append(subset)
    print(powerset)

    for behaviors_to_remove in powerset:
        behaviors_in_mat = tuple([elem for elem in behaviors if elem not in behaviors_to_remove])
        new_matrix = np.delete(outcome_matrix, behaviors_to_remove, 0)
        new_matrix = np.delete(new_matrix, behaviors_to_remove, 1)
        print(behaviors_in_mat)
        print(new_matrix)
        if len(behaviors_in_mat) > 1 and np.linalg.det(new_matrix) != 0:
            new_matrix = np.linalg.inv(new_matrix)
            result = np.matmul(new_matrix, np.ones((len(behaviors_in_mat), 1)))
            result = (result / sum(result))
            print(result)
            print(tuple(result.transpose()[0]))
            result2 = [0, 0, 0, 0, 0]
            for i, beh in enumerate(behaviors_in_mat):
                result2[beh] = result[i][0]
            if all(elem >= 0 for elem in result2):
                population = population_type(size=100000, gens_in_sim=2500,
                                             fitness_offspring_factor=0.1, random_offspring_factor=0.00,
                                             outcome_matrix=outcome_matrix,
                                             behaviors=behaviors, starting_animal_ratios=tuple(result2))
                print(np.matmul(np.linalg.inv(new_matrix), result))
                population.run_simulation()
                # plot_data(population)
        print("----------------")
