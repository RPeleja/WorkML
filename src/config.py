"""
    Central configuration management for the wine quality prediction project.
    
    Attributes:
        RANDOM_STATE (int): Seed for reproducibility (42)
        TEST_SIZE (float): Proportion of data used for testing (0.2)
        WEATHER_MONTHS (list): Months when wine data is collected [9, 10]
        MODEL_PATH (str): Directory for saving trained models
        DATA_PATH (str): Directory for input data files
"""
class Config:
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    WEATHER_MONTHS = [9, 10]  # September and October
    MODEL_PATH = 'WorkML/models/'
    DATA_PATH = 'WorkML/data/'