"""
    Central configuration management for the wine quality prediction project.
    
    Attributes:
        N_ESTIMATORS (int): Number of estimators for ensemble models (100)
        RANDOM_STATE (int): Seed for reproducibility (42)
        TEST_SIZE (float): Proportion of data used for testing (0.2)
        WEATHER_MONTHS (list): Months when wine data is collected [9, 10]
        MODEL_PATH (str): Directory for saving trained models
        DATA_PATH (str): Directory for input data files
"""
class Config:
    
    N_ESTIMATORS = 100
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    WEATHER_MONTHS = [8, 10]  # August to October
    MODEL_PATH = 'models/'
    DATA_PATH = 'data/'