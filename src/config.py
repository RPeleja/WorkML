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
    
    # Define hyperparameters to tune
    param_grid_RF = {
        'n_estimators': [50, 100, 200],  
        'max_depth': [10, 20, None],  
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4],  
        'max_features': ['sqrt', 'log2'],  
        'bootstrap': [True, False]  
    }
    
    param_grid_XB = {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depth': [3, 5, 7],  # Tree depth
        'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
        'subsample': [0.7, 0.9],  # Fraction of samples per tree
        'colsample_bytree': [0.7, 0.9],  # Fraction of features per tree
        'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required
    }
    