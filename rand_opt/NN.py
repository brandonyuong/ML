import mlrose
from time import process_time
from sklearn.metrics import accuracy_score

from rand_opt.helpers import *


def analyze_nn():

    # Load Data Set.  Make sure to run data_creation.py first
    x_train = pd.read_csv('x_train.csv')
    x_test = pd.read_csv('x_test.csv')
    y_train = pd.read_csv('y_train.csv')
    y_test = pd.read_csv('y_test.csv')

    iter_list = [100, 200, 500, 1000, 2000]
    rhc_avgs = []
    sa_avgs = []
    ga_avgs = []

    for i in iter_list:
        runs_per_iter = 10
        fit_time_rhc = [None] * runs_per_iter
        fit_time_sa = [None] * runs_per_iter
        fit_time_ga = [None] * runs_per_iter
        train_acc_rhc = [None] * runs_per_iter
        train_acc_sa = [None] * runs_per_iter
        train_acc_ga = [None] * runs_per_iter
        test_acc_rhc = [None] * runs_per_iter
        test_acc_sa = [None] * runs_per_iter
        test_acc_ga = [None] * runs_per_iter
        for j in range(runs_per_iter):
            # Initialize NN with RHC
            nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes=[12], activation='tanh',
                                                algorithm='random_hill_climb',
                                                max_iters=i,
                                                bias=True, is_classifier=True,
                                                learning_rate=0.1,
                                                early_stopping=True, clip_max=50,
                                                max_attempts=200)
            nn_model_sa = mlrose.NeuralNetwork(hidden_nodes=[12], activation='tanh',
                                               algorithm='simulated_annealing',
                                               max_iters=i,
                                               bias=True, is_classifier=True,
                                               learning_rate=0.1,
                                               early_stopping=True, clip_max=50,
                                               max_attempts=200)
            nn_model_ga = mlrose.NeuralNetwork(hidden_nodes=[12], activation='tanh',
                                               algorithm='genetic_alg', max_iters=i,
                                               bias=True, is_classifier=True,
                                               learning_rate=0.1,
                                               early_stopping=True, clip_max=50,
                                               max_attempts=200)

            fit_time_rhc[j], train_acc_rhc[j], test_acc_rhc[j] = analyze_fit(
                nn_model_rhc, x_train, x_test, y_train, y_test)
            fit_time_sa[j], train_acc_sa[j], test_acc_sa[j] = analyze_fit(
                nn_model_sa, x_train, x_test, y_train, y_test)
            fit_time_ga[j], train_acc_ga[j], test_acc_ga[j] = analyze_fit(
                nn_model_ga, x_train, x_test, y_train, y_test)
        rhc_avgs.append([avg_list(fit_time_rhc), avg_list(train_acc_rhc),
                         avg_list(test_acc_rhc)])
        sa_avgs.append([avg_list(fit_time_sa), avg_list(train_acc_sa),
                        avg_list(test_acc_sa)])
        ga_avgs.append([avg_list(fit_time_ga), avg_list(train_acc_ga),
                        avg_list(test_acc_ga)])

    rhc_avgs_arr = np.array(rhc_avgs)
    sa_avgs_arr = np.array(sa_avgs)
    ga_avgs_arr = np.array(ga_avgs)
    print("RHC (Time, Train, Test): ", rhc_avgs_arr)
    print("SA (Time, Train, Test): ", sa_avgs_arr)
    print("GA (Time, Train, Test): ", ga_avgs_arr)

    plt.figure()

    title = "NN Fit Time vs Iterations"
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.grid()
    plot_rand_opt(iter_list, rhc_avgs_arr[:, 0], plot_label="RHC", custom_color="#f92672")
    plot_rand_opt(iter_list, sa_avgs_arr[:, 0], plot_label="SA", custom_color="#007fff")
    plot_rand_opt(iter_list, ga_avgs_arr[:, 0], plot_label="GA", custom_color="#05acbf")
    plt.legend(loc="best")
    plt.savefig(title + ".png")

    plt.clf()

    title = "NN Training Accuracy vs Iterations"
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.grid()
    plot_rand_opt(iter_list, rhc_avgs_arr[:, 1], plot_label="RHC", custom_color="#f92672")
    plot_rand_opt(iter_list, sa_avgs_arr[:, 1], plot_label="SA", custom_color="#007fff")
    plot_rand_opt(iter_list, ga_avgs_arr[:, 1], plot_label="GA", custom_color="#05acbf")
    plt.legend(loc="best")
    plt.savefig(title + ".png")

    plt.clf()

    title = "NN Test Accuracy vs Iterations"
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.grid()
    plot_rand_opt(iter_list, rhc_avgs_arr[:, 2], plot_label="RHC", custom_color="#f92672")
    plot_rand_opt(iter_list, sa_avgs_arr[:, 2], plot_label="SA", custom_color="#007fff")
    plot_rand_opt(iter_list, ga_avgs_arr[:, 2], plot_label="GA", custom_color="#05acbf")
    plt.legend(loc="best")
    plt.savefig(title + ".png")

    plt.close()


# return running time, train accuracy, test accuracy
def analyze_fit(nn_model, x_train, x_test, y_train, y_test):
    start_time = process_time()
    nn_model.fit(x_train, y_train)
    running_time = process_time() - start_time
    print('Running Time: ', running_time)

    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model.predict(x_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print('Training accuracy: ', y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model.predict(x_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    print('Test accuracy: ', y_test_accuracy)

    return running_time, y_train_accuracy, y_test_accuracy


if __name__ == '__main__':
    analyze_nn()
