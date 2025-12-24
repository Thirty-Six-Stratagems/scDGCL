import numpy as np
import math
import pandas as pd
import os
from Cluster import Cluster


class SSA:
    def __init__(self, data, data_label=None, k=None, max_iter=60, RESULT_PATH="./result"):
        self.max_iter = max_iter
        self.data = data
        self.cell = data.shape[0]
        self.gene = data.shape[1]
        self.k = k
        self.label = data_label

        self.Cluster = Cluster(self.data, self.label, self.k)

        self.Pdp = 0.1
        self.Gc = 2
        self.dg = lambda: np.random.uniform(0.3, 0.7)

        self.sc_list = []
        self.fitness_list = []
        self.feature_list = []
        self.subset_index = []
        self.last_fitness = None
        self.unchanged_count = 0
        self.stop_threshold = 12

        self.RESULT_PATH = RESULT_PATH

    def random_location(self):
        return np.random.uniform(0, 1, self.gene) * 2 * np.pi

    def init_population(self, squirrel_num=50):
        pop = np.zeros([squirrel_num, self.gene], dtype=np.float64)
        for i in range(squirrel_num):
            pop[i] = self.random_location()
        return pop

    def run_task(self):
        population = self.init_population()
        for i in range(self.max_iter):
            population = self.task(population=population, epoch=i)
            if self.check_convergence():
                print(f"Converged after {i + 1} iterations.")
                break

        self.cluster(population, i + 1)
        self.save_results()
        return self.data[:, self.subset_index]

    def task(self, population, epoch):
        indices = self.cluster(population, epoch)

        ht = indices[0]
        at = indices[1:4]
        nt = indices[4:]
        nt_1 = nt[[x for x in range(46) if x % 2 == 0]]
        nt_2 = nt[[x for x in range(46) if x % 2 != 0]]

        new_pop = population.copy()

        for index in at:
            if np.random.random() >= self.Pdp:
                new_pop[index] = new_pop[index] + self.dg() * self.Gc * (new_pop[ht] - new_pop[index])
            else:
                new_pop[index] = self.random_location()

        for index in nt_1:
            if np.random.random() >= self.Pdp:
                new_pop[index] = new_pop[index] + self.dg() * self.Gc * (new_pop[np.random.choice(at)] - new_pop[index])
            else:
                new_pop[index] = self.random_location()

        for index in nt_2:
            if np.random.random() >= self.Pdp:
                new_pop[index] = new_pop[index] + self.dg() * self.Gc * (new_pop[ht] - new_pop[index])
            else:
                new_pop[index] = self.random_location()

        s_min = 1e-5 / (365 ** ((epoch + 1) / (self.max_iter / 2.5)))
        sc = np.sqrt(np.sum((new_pop[at] - new_pop[ht]) ** 2))
        if sc < s_min:
            new_pop[nt_1] = 0 + self.levy_flight(size=population.shape[1]) * (2 * np.pi - 0)

        return new_pop

    def cluster(self, population, epoch):
        fitness, ari, feature, subset_index_list = self.Cluster.fitness_kmeans(population)

        indices = np.argsort(-np.array(fitness))

        best_fitness = fitness[indices[0]]
        best_sc = ari[indices[0]]
        best_feature = feature[indices[0]]

        print("epoch {0} : fitness = {1} , sc = {2} , feature = {3}".format(
            epoch, best_fitness, best_sc, best_feature))

        self.fitness_list.append(best_fitness)
        self.feature_list.append(best_feature)
        self.sc_list.append(best_sc)
        self.subset_index = subset_index_list[indices[0]]

        self.update_convergence_check(best_fitness)

        return indices

    def update_convergence_check(self, current_fitness):
        if self.last_fitness is None:
            self.last_fitness = current_fitness
        elif current_fitness == self.last_fitness:
            self.unchanged_count += 1
        else:
            self.last_fitness = current_fitness
            self.unchanged_count = 0

    def check_convergence(self):
        return self.unchanged_count >= self.stop_threshold

    def save_results(self):
        if not os.path.exists(self.RESULT_PATH):
            os.makedirs(self.RESULT_PATH)

        subset = self.data[:, self.subset_index]
        subset_path = os.path.join(self.RESULT_PATH, "subset.csv")
        pd.DataFrame(subset).to_csv(subset_path)

    def levy_flight(self, alpha=0.01, beta=1.5, size=None):
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        sample = alpha * u / (np.abs(v) ** (1 / beta))
        return sample