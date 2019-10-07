import mlrose
from time import process_time
import random

from rand_opt.helpers import *


def run_ff_rhc(arr_len, max_iters):
    fitness = mlrose.FlipFlop()
    problem = mlrose.DiscreteOpt(length=arr_len, fitness_fn=fitness,
                                 maximize=True)

    start_time = process_time()
    state, fit, curve = mlrose.random_hill_climb(problem, max_iters=max_iters)
    running_time = process_time() - start_time

    print('*** RHC ***')
    print('Running Time: ', running_time)
    print('The best state found is: ', state)
    print('The fitness at the best state is: ', fit)
    return running_time, fit


def run_ff_sa(arr_len, max_iters):
    fitness = mlrose.FlipFlop()
    problem = mlrose.DiscreteOpt(length=arr_len, fitness_fn=fitness,
                                 maximize=True)

    start_time = process_time()
    state, fit, curve = mlrose.simulated_annealing(problem, max_iters=max_iters)
    running_time = process_time() - start_time

    print('*** SA ***')
    print('Running Time: ', running_time)
    print('The best state found is: ', state)
    print('The fitness at the best state is: ', fit)
    return running_time, fit


def run_ff_ga(arr_len, max_iters):
    fitness = mlrose.FlipFlop()
    problem = mlrose.DiscreteOpt(length=arr_len, fitness_fn=fitness,
                                 maximize=True)

    start_time = process_time()
    state, fit, curve = mlrose.genetic_alg(problem, max_iters=max_iters)
    running_time = process_time() - start_time

    print('*** GA ***')
    print('Running Time: ', running_time)
    print('The best state found is: ', state)
    print('The fitness at the best state is: ', fit)
    return running_time, fit


def run_ff_mimic(arr_len, max_iters):
    fitness = mlrose.FlipFlop()
    problem = mlrose.DiscreteOpt(length=arr_len, fitness_fn=fitness,
                                 maximize=True)

    start_time = process_time()
    state, fit, curve = mlrose.mimic(problem, max_iters=max_iters)
    running_time = process_time() - start_time

    print('*** MIMIC ***')
    print('Running Time: ', running_time)
    print('The best state found is: ', state)
    print('The fitness at the best state is: ', fit)
    return running_time, fit


iter_list = [50, 100, 500, 1000, 2000, 4000, 8000]

# Run for each total number of items
for i in range(3):
    if i == 0:
        arr_length = 10
    elif i == 1:
        arr_length = 50
    else:
        arr_length = 100

    rhc_avgs = [None] * len(iter_list)
    sa_avgs = [None] * len(iter_list)
    ga_avgs = [None] * len(iter_list)
    mimic_avgs = [None] * len(iter_list)

    # index counter for adding to avgs lists above
    counter = 0

    # Compute for each iter value
    for j in iter_list:
        runs_per_iter = 3
        fit_time_rhc = [None] * runs_per_iter
        fit_time_sa = [None] * runs_per_iter
        fit_time_ga = [None] * runs_per_iter
        fit_time_mimic = [None] * runs_per_iter
        fit_rhc = [None] * runs_per_iter
        fit_sa = [None] * runs_per_iter
        fit_ga = [None] * runs_per_iter
        fit_mimic = [None] * runs_per_iter

        # Run 3 times each
        for k in range(runs_per_iter):
            fit_time_rhc[k], fit_rhc[k] = run_ff_rhc(arr_length, j)
            fit_time_sa[k], fit_sa[k] = run_ff_sa(arr_length, j)
            fit_time_ga[k], fit_ga[k] = run_ff_ga(arr_length, j)
            fit_time_mimic[k], fit_mimic[k] = run_ff_mimic(arr_length, j)

        rhc_avgs[counter] = [avg_list(fit_time_rhc), avg_list(fit_rhc)]
        sa_avgs[counter] = [avg_list(fit_time_sa), avg_list(fit_sa)]
        ga_avgs[counter] = [avg_list(fit_time_ga), avg_list(fit_ga)]
        mimic_avgs[counter] = [avg_list(fit_time_mimic), avg_list(fit_mimic)]
        counter += 1

    print("RHC (Time, Train, Test): ", rhc_avgs)
    print("SA (Time, Train, Test): ", sa_avgs)
    print("GA (Time, Train, Test): ", ga_avgs)
    print("MIMIC (Time, Train, Test): ", mimic_avgs)
    name = "Flip Flop: " + str(arr_length) + " Length Results.txt"
    with open(name, "w", newline="") as f:
        f.write("RHC: ")
        for item in rhc_avgs:
            f.write(str(item))
        f.write("\nSA: ")
        for item in sa_avgs:
            f.write(str(item))
        f.write("\n GA: ")
        for item in ga_avgs:
            f.write(str(item))
        f.write("\nMIMIC: ")
        for item in mimic_avgs:
            f.write(str(item))

    rhc_avgs_arr = np.array(rhc_avgs)
    sa_avgs_arr = np.array(sa_avgs)
    ga_avgs_arr = np.array(ga_avgs)
    mimic_avgs_arr = np.array(mimic_avgs)

    plt.figure()

    xi = list(range(len(iter_list)))

    title = "Flip Flop: " + str(arr_length) + " Length Fit Times"
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Fit Time (s)")
    plt.xticks(xi, iter_list)
    plot_rand_opt(xi, rhc_avgs_arr[:, 0], plot_label="RHC", custom_color="#f92672")
    plot_rand_opt(xi, sa_avgs_arr[:, 0], plot_label="SA", custom_color="#007fff")
    plot_rand_opt(xi, ga_avgs_arr[:, 0], plot_label="GA", custom_color="#05acbf")
    plot_rand_opt(xi, mimic_avgs_arr[:, 0], plot_label="MIMIC", custom_color="#b5b9fc")
    plt.grid()
    plt.legend(loc="best")
    plt.savefig(title + ".png")

    plt.clf()

    title = "Flip Flop: " + str(arr_length) + " Length Fit"
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.xticks(xi, iter_list)
    plot_rand_opt(xi, rhc_avgs_arr[:, 1], plot_label="RHC", custom_color="#f92672")
    plot_rand_opt(xi, sa_avgs_arr[:, 1], plot_label="SA", custom_color="#007fff")
    plot_rand_opt(xi, ga_avgs_arr[:, 1], plot_label="GA", custom_color="#05acbf")
    plot_rand_opt(xi, mimic_avgs_arr[:, 1], plot_label="MIMIC", custom_color="#b5b9fc")
    plt.grid()
    plt.legend(loc="best")
    plt.savefig(title + ".png")

    plt.close()
