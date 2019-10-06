import mlrose
import csv
from time import process_time

from rand_opt.helpers import *
from rand_opt.fastmimic import mlrose as fastmimic


coords_list = []
with open('TSP_10_coords.csv', 'r') as filehandle:
    for line in filehandle:
        coords_list.append(tuple(map(int, line.strip().split(','))))
print(coords_list)

problem = mlrose.TSPOpt(length=len(coords_list), coords=coords_list,
                        maximize=False)

# Solve problem using the genetic algorithm
state, fit, curve = mlrose.genetic_alg(problem, mutation_prob=0.2, random_state=2)

print('The best state found is: ', state)
print('The fitness at the best state is: ', fit)
