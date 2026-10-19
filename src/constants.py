
# =======================================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# =======================================================================

DATASET_PATH = "datasets/dataset_cell_growth_4factor.csv"

FEATURES = [
    "Temperature [C]",
    "pH",
    "Glucose [mM]",
    "Lactate [mM]"
]

TARGET = "Specific Growth Rate [d^-1]"

TEST_SIZE = 0.2
RANDOM_STATE = 42
