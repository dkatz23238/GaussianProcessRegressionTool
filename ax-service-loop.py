from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import branin

from dgprt import main, plot_results
from numpy import mean, abs, sqrt

import datetime
from os import mkdir
from os.path import exists, join

import ax.storage as axst
import os

from parameters import PARAMETERS

isonow = lambda : datetime.datetime.now().isoformat()

def evaluate(parameters):
    LEARNING_RATE = parameters["LEARNING_RATE"]
    ROLLING_VAL = parameters["ROLLING_VAL"]
    N1_LAYERS = parameters["N1_LAYERS"]
    N2_LAYERS = parameters["N2_LAYERS"]
    N_EPOCHS = parameters["N_EPOCHS"]
    GRID_SIZE = parameters["GRID_SIZE"]

    results, _ = main(LEARNING_RATE, ROLLING_VAL, N1_LAYERS,
                   N2_LAYERS, N_EPOCHS, GRID_SIZE)
    
    mse = sqrt(mean((results.predicted - results.y_true)**2))
    return float(mse)

ax = AxClient()

N_TRIALS = int(os.environ.get("N_TRIALS", 5))

ax.create_experiment(
    name="GaussianProcessRegression-%s" % isonow(),
    parameters=PARAMETERS,
    objective_name="mean_square_error",
    minimize=True)

for _ in range(N_TRIALS):
    print(f"[ax-service-loop] Trial {_+1} of {N_TRIALS}")
    parameters, trial_index = ax.get_next_trial()
    ax.complete_trial(
        trial_index=trial_index,
        raw_data= evaluate(parameters)
    )
    print(parameters)
    print("")


print("[ax-service-loop] Training complete!")
best_parameters, metrics = ax.get_best_parameters()
print(f"[ax-service-loop] Sending data to db.")
print(f"[ax-service-loop] Best parameters found: {best_parameters}")
DB_URL = os.environ.get("DB_URL", "mysql://root:password@localhost/axdb")
from sqlalchemy import create_engine
engine = axst.sqa_store.db.create_mysql_engine_from_url(url=DB_URL)
conn = engine.connect()
axst.sqa_store.db.init_engine_and_session_factory(url=DB_URL)
table_names = engine.table_names()
axst.sqa_store.db.create_all_tables(engine)
axst.sqa_store.save(experiment=ax.experiment)
conn.close()
engine.dispose()

