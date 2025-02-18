"""
    Handles all data preprocessing steps including scaling and imputation.
    
    Attributes:
        scaler: MinMaxScaler instance for feature scaling
        imputer: SimpleImputer instance for handling missing values
    
    Methods:
        preprocess(df):
            Performs initial data cleaning and feature engineering
            - Removes unnecessary columns
            - Extracts time features
            - Creates target variable
            Returns: Preprocessed DataFrame
            
        prepare_features(df):
            Separates features and target variables
            Returns: (features DataFrame, target Series)
            
        fit_transform(X_train, X_test):
            Applies scaling and imputation to training and test data
            Returns: (transformed X_train, transformed X_test)
    
    Usage:
        preprocessor = DataPreprocessor()
        df = preprocessor.preprocess(raw_df)
        X, y = preprocessor.prepare_features(df)
        X_train_scaled, X_test_scaled = preprocessor.fit_transform(X_train, X_test)
"""

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE),
            'xgboost': XGBRegressor(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE),
            'svm': SVR()
        }

    def split_data(self, X, y):
        return train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )

    def train_models(self, X_train, y_train, X_test, y_test):
        trained_models = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test,y_test)  # R² Score
            trained_models[name] = model
            print(f"{name} trained. Test R² Score: {score:.4f}")
        return trained_models
    
    def tune_model_RF(self, X_train, y_train):
        
        # Initialize Random Forest Regressor
        rf = RandomForestRegressor(random_state=42)

        # Use RandomizedSearchCV for efficiency
        grid_search = RandomizedSearchCV(
            rf, self.config.param_grid, cv=5, scoring='neg_mean_absolute_error', 
            n_iter=10, n_jobs=-1, verbose=2, random_state=42
        )
        
        grid_search.fit(X_train, y_train)

        # Best parameters & score
        print("Best Params:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)

        # Train best model
        best_rf = RandomForestRegressor(
            n_estimators=grid_search.best_params_['n_estimators'],
            max_depth=grid_search.best_params_['max_depth'],
            min_samples_split=grid_search.best_params_['min_samples_split'],
            min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
            max_features=grid_search.best_params_['max_features'],
            bootstrap=grid_search.best_params_['bootstrap'],
            random_state=42
        )

        best_rf.fit(X_train, y_train)
        return best_rf