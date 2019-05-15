from dgprt import main, plot_results
from numpy import mean, abs

import datetime
from os import mkdir
from os.path import exists, join

isonow = lambda : datetime.datetime.now().isoformat()

def save_results(results, root_dir):
    if not exists(join(root_dir, "experiments-results")):
        mkdir(join(root_dir, "experiments-results"))

    path = join(root_dir, "experiments-results", isonow()+".csv" )
    results.to_csv(path, encoding="utf-8")


if __name__ == "__main__":

    RESULTS_ROOT_PATH = "."
    LEARNING_RATE = 0.10
    ROLLING_VAL = 10
    N1_LAYERS = 100
    N2_LAYERS = 50
    N_EPOCHS = 25
    GRID_SIZE = 100

    results = main(LEARNING_RATE, ROLLING_VAL, N1_LAYERS,
               N2_LAYERS, N_EPOCHS, GRID_SIZE)
    print(results)

    mae = mean(abs( results.predicted - results.y_true ))

    print(f"Mean Absoloute Error: {mae}")
    # Uncomment the following line to plot results:
    # plot_results(results)
    save_results(results, RESULTS_ROOT_PATH)

