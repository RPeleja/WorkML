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

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def preprocess(self, df):
        # Drop unnecessary columns
        columns_to_drop = [
            'entity_id', 'entity_type', 'name', 'latitude', 'longitude',
            'date_observed', 'time_observed', 'barometricpressure', 
            'uv_index', 'total sulfur dioxide', 'timestamp'
        ]

        # Extract time features
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month

        df = df.drop(columns=columns_to_drop)

        # Create target variable
        df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
        
        return df

    def prepare_features(self, df):
        features = df.drop(['quality', 'best quality'], axis=1)
        target = df['best quality']
        return features, target

    def fit_transform(self, X_train, X_test):
        X_train = self.imputer.fit_transform(X_train)
        X_test = self.imputer.transform(X_test)
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test