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

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {
            'logistic': LogisticRegression(),
            'xgboost': XGBClassifier(),
            'svm': SVC(kernel='rbf', probability=True),
            'random_forest': RandomForestClassifier(
                n_estimators=config.N_ESTIMATORS, 
                random_state=config.RANDOM_STATE
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=config.N_ESTIMATORS, 
                random_state=config.RANDOM_STATE
            )
        }

    def split_data(self, X, y):
        return train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )

    def train_models(self, X_train, y_train):
        trained_models = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
        return trained_models