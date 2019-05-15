PARAMETERS = [
    {
        "name": "LEARNING_RATE",
        "type": "range",
        "bounds": [0.001, 0.1],
        "value_type": "float",
    },
    {
        "name": "ROLLING_VAL",
        "type": "range",
        "bounds": [10, 30],
        "value_type": "int",
    },
    {
        "name": "N1_LAYERS",
        "type": "range",
        "bounds": [10, 100],
        "value_type": "int",
    },
    {
        "name": "N2_LAYERS",
        "type": "range",
        "bounds": [10, 100],
        "value_type": "int",
    },
    {
        "name": "N_EPOCHS",
        "type": "fixed",
        "value": 5,
        "value_type": "int",
    },
    {
        "name": "GRID_SIZE",
        "type": "range",
        "bounds": [50, 200],
        "value_type": "int",
    },
]