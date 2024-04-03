import json
import random
import math
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import mode
from sklearn.cluster import KMeans
from numpy.random import choice, rand
import logging
import abc
import pprint
import numpy
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List
import time


rng = numpy.random.default_rng(2023)
numpy.random.seed(2023)
random.seed(2023)


class VRP:

    def __init__(
        self,
        locations: list[list[float]],
        num_salesmen: int,
        precompute_distances: bool = True,
    ):
        # Set the given locations (with depot at index 0)
        self.locations: list[list[float]] = locations
        # Set the total number of cities to visit
        self.num_locations: int = len(locations)
        # Set the given number of salesmen that should be coordinated and routed between the cities
        self.num_salesmen: int = num_salesmen
        # Pre-compute the distances between the given cities
        if precompute_distances is True:
            self._distance_matrix = self._precompute_distances()
        else:
            self._distance_matrix = None
        # Validate the given input
        self._validate()

    def _validate(self):
        # Make sure that at least one depot and a city is given in the list
        if len(self.locations) < 2:
            raise ValueError()
        else:
            # Make sure that a tuple of coordinates is given for each location
            for pair in self.locations:
                if len(pair) != 2:
                    raise ValueError()
        # Make sure that at least one salesman can be routed between the locations
        if self.num_salesmen < 1:
            raise ValueError()

    def _precompute_distances(self) -> list[list[float]]:
        return [
            [
                numpy.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
                for c1 in self.locations
            ]
            for c2 in self.locations
        ]

    def _distance(p1: float, p2: float):
        return numpy.sqrt(numpy.sum(numpy.power(p2 - p1, 2)))

    def distance(self, loc_a: int, loc_b: int) -> float:
        if self._distance_matrix is not None:
            return self._distance_matrix[city_a][city_b]
        else:
            # return numpy.sqrt(
            #     (self.locations[loc_a][0] - self.locations[loc_b][0]) ** 2 + \
            #     (self.locations[loc_a][1] - self.locations[loc_b][1]) ** 2
            # )
            self._distance(
                p1=numpy.array(self.locations[loc_a]),
                p2=numpy.array(self.locations[loc_b]),
            )


class Individual:

    def __init__(self, chromosome: list[list[int]], generation: int):
        # Solution representation:
        # - Assume that a multipart chromosome is used
        self.chromosome: list[list[int]] = chromosome
        # The corresponding fitness value of the solution
        self.fitness: None | float = None
        self.generation: int = generation

    def pprint(self):
        # Explicitly write out the chomosome when the 'Individual' object is printed
        string = super().__str__()
        if self.chromosome is not None:
            string = string + " = \n" + 0 * " " + "[\n"
            for part in self.chromosome:
                string += 4 * " " + str(part) + ",\n"
            string += 0 * " " + "]"
            return string
        else:
            return string + " = None"

    def _validate(self):
        if self.chromosome is not None:
            if len(self.chromosome) > 0:
                raise ValueError(f"{self.chromosome}")


class BaseFitnessFunction(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def evaluate(self, individual: Individual):
        pass


class BasePopulationInitializer(abc.ABC):

    def __init__(
        self,
        population_size: int,
        vrp_instance: VRP,
        fitness_function_instance: BaseFitnessFunction,
    ):
        self.population_size: int = population_size
        self.vrp_instance: VRP = vrp_instance
        self.fitness_function_instance: BaseFitnessFunction = fitness_function_instance

    @abc.abstractmethod
    def generate(self):
        pass


class BaseSolver(abc.ABC):

    def __init__(
        self,
        vrp_instance: VRP,
        population_size: int,
        population_initializer_class: BasePopulationInitializer,
        fitness_function_class: BaseFitnessFunction,
    ):
        self.vrp_instance = vrp_instance
        self.population_size: int = population_size
        # Instantiate fitness function class
        self.fitness_function_instance = fitness_function_class(
            vrp_instance=vrp_instance,
        )
        # Instantiate solution initializer class
        self.population_initializer_instance = population_initializer_class(
            population_size=self.population_size,
            vrp_instance=vrp_instance,
            fitness_function_instance=self.fitness_function_instance,
        )
        # self.population: None | Population = None
        # self.solution_history: list = [] # TODO
        self._validate()

    def _validate(self):
        if self.population_size < 1:
            raise ValueError(f"{self.population_size}")

    @abc.abstractmethod
    def _initialization(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass


class Population:

    def __init__(
        self, individuals: list[Individual]
    ):  # , fitness_function_instance: "BaseFitnessFunction"):
        self.individuals: list[Individual] = individuals
        # self.fitness_function_instance: BaseFitnessFunction = fitness_function_instance

    # def evaluate(self):
    #     for individual in self.individuals:
    #         individual.fitness = self.fitness_function_instance.evaluate(individual)

    def pprint(self):
        string = ""
        for inividual in self.individuals:
            string += inividual.__str__() + "\n"
        return string

    def __len__(self) -> int:
        return len(self.individuals)

    def __getitem__(self, item) -> Individual:
        return self.individuals[item]

    def __add__(self, other_population):
        return Population(self.individuals + other_population.individuals)

    def random_pick(self) -> Individual:
        return random.choice(self.individuals)

    def size(self) -> int:
        return len(self.individuals)

    def sort(self, reverse: bool = False):
        sorted_population = sorted(
            self.individuals,
            key=lambda x: x.fitness,
            reverse=reverse,
        )
        self.individuals = sorted_population

    def prune(self, population_size: int, reverse: bool = False):
        self.sort(reverse=reverse)
        self.individuals = self.individuals[:population_size]

    def get_topk(self, k: int = 1) -> list[Individual]:
        return self.individuals[:k]


def route_cost(vrp_instance: VRP, route: list[int]):
    total_distance = 0.0
    if len(route) == 0:
        # distance = numpy.inf
        distance = 0.0
    else:
        # Distance from depot to first stop + distance from last to to depot
        distance = (
            vrp_instance._distance_matrix[0][route[0]]
            + vrp_instance._distance_matrix[route[-1]][0]
        )
        # All distances in between
        for i in range(1, len(route)):
            distance += vrp_instance._distance_matrix[route[i - 1]][route[i]]
    total_distance += distance
    return total_distance


class TwoOptSolver(BaseSolver):

    def __init__(
        self,
        vrp_instance: VRP,
        population_size: int,
        population_initializer_class: BasePopulationInitializer,
        fitness_function_class: BaseFitnessFunction,
    ):
        super().__init__(
            vrp_instance,
            population_size,
            population_initializer_class,
            fitness_function_class,
        )

    def _initialization(self):
        return self.population_initializer_instance.generate()

    def two_opt(self, individual: Individual, improvement_threshold: float = 0.001):
        # Reverse the order of all elements from element i to element k in array r.
        # two_opt_swap = lambda r, i, k: numpy.concatenate((r[0:i], r[k:-len(r)+i-1:-1], r[k+1:len(r)]))
        # two_opt_swap = lambda r, i, k: r[0:i] + r[k:-len(r)+i-1:-1] + r[k+1:len(r)]
        def two_opt_swap(route, i, k):
            # return numpy.concatenate((route[:i], route[i:k+1][::-1], route[k+1:]))
            return route[:i] + route[i : k + 1][::-1] + route[k + 1 :]

        _routes = []
        for route in individual.chromosome:
            if len(route) > 0:
                _route = route.copy()
                best_cost = route_cost(self.vrp_instance, _route)
                improvement_factor = 1
                while improvement_factor > improvement_threshold:
                    cost_to_beat = best_cost
                    for swap_first in range(0, len(_route) - 1):
                        for swap_last in range(swap_first + 1, len(_route)):
                            new_route = two_opt_swap(_route, swap_first, swap_last)
                            new_cost = route_cost(self.vrp_instance, new_route)
                            if new_cost < best_cost:
                                _route = new_route
                                best_cost = new_cost
                    improvement_factor = 1 - best_cost / cost_to_beat
            _routes.append(_route)
        individual = Individual(
            chromosome=_routes,
            generation=0,
        )
        return individual

    def run(self):
        population = self._initialization()
        population.sort(reverse=True)
        # return population
        individuals = []
        for individual in population.individuals:
            individual = self.two_opt(individual=individual)
            individuals.append(individual)
        individuals = [
            self.fitness_function_instance.evaluate(individual)
            for individual in individuals
        ]

        population = Population(individuals=individuals)
        population.sort(reverse=True)
        return population


class FitnessFunctionMinimizeDistance(BaseFitnessFunction):

    def __init__(self, vrp_instance: VRP):
        super().__init__()
        self.vrp_instance: VRP = vrp_instance

    def evaluate(self, individual: Individual) -> Individual:
        total_distance = 0.0
        for route in individual.chromosome:
            if len(route) == 0:
                # distance = numpy.inf
                distance = 0.0
            else:
                # Distance from depot to first stop + distance from last to to depot
                distance = (
                    self.vrp_instance._distance_matrix[0][route[0]]
                    + self.vrp_instance._distance_matrix[route[-1]][0]
                )
                # All distances in between
                for i in range(1, len(route)):
                    distance += self.vrp_instance._distance_matrix[route[i - 1]][
                        route[i]
                    ]
            total_distance += distance
        # return total_distance
        if total_distance == 0.0:
            individual.fitness = 0.0
        else:
            individual.fitness = 1.0 / total_distance
        return individual


class RandomPopulationInitializer(BasePopulationInitializer):

    def __init__(
        self,
        population_size: int,
        vrp_instance: VRP,
        fitness_function_instance: BaseFitnessFunction,
    ):
        super().__init__(
            population_size=population_size,
            vrp_instance=vrp_instance,
            fitness_function_instance=fitness_function_instance,
        )

    def generate(self) -> Population:
        individuals = [self._create_individual() for _ in range(self.population_size)]
        individuals = [
            self.fitness_function_instance.evaluate(individual)
            for individual in individuals
        ]
        return Population(individuals=individuals)

    def _create_individual(self):
        route = list(range(1, self.vrp_instance.num_locations))
        random.shuffle(route)
        partition_points = sorted(
            random.sample(route[:-1], self.vrp_instance.num_salesmen - 1)
        )
        individual = Individual(
            chromosome=[
                route[i:j]
                for i, j in zip([0] + partition_points, partition_points + [None])
            ],
            generation=0,
        )
        return individual


class KMeansRadomizedPopulationInitializer(BasePopulationInitializer):

    def __init__(
        self,
        population_size: int,
        vrp_instance: VRP,
        fitness_function_instance: BaseFitnessFunction,
    ):
        super().__init__(
            population_size=population_size,
            vrp_instance=vrp_instance,
            fitness_function_instance=fitness_function_instance,
        )

    def generate(self) -> Population:
        kmeans_result = KMeans(
            n_clusters=self.vrp_instance.num_salesmen,
            random_state=2023,
        ).fit(self.vrp_instance.locations[1:])
        individuals = [
            self._create_individual(kmeans_result.labels_)
            for _ in range(self.population_size)
        ]
        individuals = [
            self.fitness_function_instance.evaluate(individual)
            for individual in individuals
        ]
        return Population(individuals=individuals)

    def _create_individual(self, labels):
        routes = []
        for i in range(self.vrp_instance.num_salesmen):
            route = [index + 1 for index, label in enumerate(labels) if label == i]
            random.shuffle(route)
            routes.append(route)
        individual = Individual(
            chromosome=routes,
            generation=0,
        )
        return individual


def parse_and_transform(tree):
    # Function to traverse the tree and collect nodes by depth
    def traverse(node, depth=0, nodes_by_depth={}):
        if not node:
            return
        # Handle leaf node (single value)
        if isinstance(node, int):
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)
        # Handle node with children
        else:
            value, left, right = (
                node[0],
                node[1] if len(node) > 1 else None,
                node[2] if len(node) > 2 else None,
            )
            # Process left subtree
            if left is not None:
                traverse(left, depth + 1, nodes_by_depth)
            # Process current node
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(value)
            # Process right subtree
            if right is not None:
                traverse(right, depth + 1, nodes_by_depth)

    nodes_by_depth = {}
    traverse(tree, 0, nodes_by_depth)

    # Flatten nodes_by_depth into the final list
    output = []
    for depth in sorted(nodes_by_depth.keys(), reverse=True):
        output.extend(nodes_by_depth[depth])

    return output


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def individual_to_routes(individual: Individual, vrp_instance: VRP):
    # Helper function to transform a sequence of locations (identifiers)
    # into a sequence of geospatial (latitude and longitude) coordinates
    routes: list[list[float]] = []
    for gene in individual.chromosome:
        route: list[float] = []
        # Add depot as start location
        route.append(vrp_instance.locations[0])
        for index in gene:
            route.append(vrp_instance.locations[index])
        # Add depot as end location
        route.append(vrp_instance.locations[0])
        routes.append(route)
    return routes


def plot_salesmen_routes(rotues):
    """
    Plots the routes of salesmen.

    Parameters:
    - routes: A list of lists, where each sublist contains tuples of lat/long coordinates for a salesman's tour.
    """
    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Assign a different color for each salesman's route
    colors = ["r", "g", "b", "c", "m", "y", "k"]

    for i, route in enumerate(routes):
        # Ensure we cycle through colors if there are more salesmen than colors
        color = colors[i % len(colors)]

        # Separate the latitudes and longitudes for plotting
        lats, longs = zip(*route)

        # Plot the route
        plt.plot(
            longs, lats, marker="o", color=color, linestyle="-", label=f"Salesman {i+1}"
        )

        # Optionally, you can also plot the start and end points differently
        plt.scatter(
            longs[0],
            lats[0],
            color=color,
            s=100,
            label=f"Start {i+1}",
            edgecolors="black",
        )
        plt.scatter(
            longs[-1],
            lats[-1],
            color=color,
            s=100,
            label=f"End {i+1}",
            edgecolors="black",
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Salesmen Tours")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Central location (Bissau)
    # center_latitude = 11.845729112251885
    # center_longitude = -15.5955092933175
    center_latitude = 11.852848336808085
    center_longitude = -15.598465762669719

    # # Parameters: 10 km radius, 100 sample points
    # sample_radius = 1.5  # in kilometers
    # num_samples = 50

    # seed = 123
    # # Generate random points
    # random.seed(seed)
    # # random_points = generate_random_points(center_latitude, center_longitude, sample_radius, num_samples)

    # # print(random_points)

    # # Path to save the JSON file
    file_path = "./random_geolocations.json"

    # # Write points to JSON
    # # write_points_to_json(file_path, random_points)
    # # print(f"Random geospatial locations saved to {file_path}")

    # # Read points from JSON
    points = read_points_from_json(file_path)

    # # print(points)
    points = pd.DataFrame(points)
    # salesmen = 3

    # initial_kmeans_solution(points, [center_latitude, center_longitude], salesmen, seed)
    locations = [[center_latitude, center_longitude]] + points[
        ["latitude", "longitude"]
    ].to_numpy().tolist()

    vrp_instance = VRP(
        locations=locations,
        num_salesmen=3,
        precompute_distances=True,
    )
    solver = TwoOptSolver(
        vrp_instance=vrp_instance,
        population_size=25,
        population_initializer_class=KMeansRadomizedPopulationInitializer,
        # population_initializer_class=RandomPopulationInitializer,
        fitness_function_class=FitnessFunctionMinimizeDistance,
    )

    result = solver.run()
    best_solution = result.get_topk(k=1)[0]
    print("Fitness: ", best_solution.fitness)
    routes = individual_to_routes(best_solution, vrp_instance)
    plot_salesmen_routes(routes)

    # Path to save the HTML map file
    # map_file_path = './geospatial_map.html'

    # Display points on the map
    # display_points_on_map(points, center_latitude, center_longitude, map_file_path)

    # print(f"Map with geospatial points saved to {map_file_path}"
