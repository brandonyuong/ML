import csv
import random


def tsp_coords_creation(num):
    name = "TSP_" + str(num) + "_coords.csv"
    ret_set = set()

    while len(ret_set) < num:
        ret_set.add((random.randint(1, num), random.randint(1, num)))
    print(ret_set)

    with open(name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(list(ret_set))


if __name__ == '__main__':
    tsp_coords_creation(50)
