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
    WEATHER_MONTHS = [2, 12]  # August to October
    MODEL_PATH = 'WorkML/models/'
    DATA_PATH = 'WorkML/data/'
    
    # Define hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100, 200],  
        'max_depth': [10, 20, None],  
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4],  
        'max_features': ['sqrt', 'log2'],  
        'bootstrap': [True, False]  
    }