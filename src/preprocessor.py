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

from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.scalerStd = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def preprocess(self, df):

        # The balance of sulfur dioxide and sugar-acidity affects wine preservation & taste.
        # Helps reveal nonlinear relationships.
        df['sulfur_ratio'] = df['free_sulfur_dioxide'] / (df['total_sulfur_dioxide'] + 1e-6)
        df['sugar_acidity_ratio'] = df['residual_sugar'] / (df['fixed_acidity'] + 1e-6)
        
        # Interaction Features
        df["alcohol_sulphates"] = df["alcohol"] * df["sulphates"]
        df["density_pH"] = df["density"] * df["pH"]
        df["sulphates_acidity"] = df["sulphates"] * df["volatile_acidity"]
        df["sugar_acidity"] = df["residual_sugar"] * df["fixed_acidity"]
        
        # Convert precipitation into a binary feature (rain vs. no rain)
        df['rainy'] = (df['precipitation'] > 0).astype(int)
        
        # Fill missing values in windspeed with the mean
        mean_windspeed = df['windspeed'].mean()
        df['windspeed'].fillna(mean_windspeed, inplace=True)

        # Normalize windspeed & temperature (staprecipitationndardize between 0 and 1)
        df[['windspeed', 'temperature']] = self.scaler.fit_transform(df[['windspeed', 'temperature']])
        
        # Create target variable
        df['best_quality'] = [1 if x > 6 else 0 for x in df.quality]
        
        # Drop unnecessary columns
        columns_to_drop = [
            'entity_type', 'latitude', 'longitude', 'hour', 'month', 'precipitation',
            'date_observed', 'time_observed', 'barometricpressure', 
            'uv_index', 'total_sulfur_dioxide', 'timestamp', 'uvindexmax', 'solarradiation'
        ]

        df = df.drop(columns=columns_to_drop)

        return df

    def prepare_features(self, df):
        features = df.drop(['quality', 'best_quality'], axis=1)
        target = df['quality']
        return features, target

    def fit_transform(self, X_train, X_test, bool):
        
        if bool:
            X_train = self.imputer.fit_transform(X_train)
            X_test = self.imputer.transform(X_test)
            
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        return X_train, X_test
    
    def fit(self, X_train):
        """Fit the imputer and scaler using training data only"""
        X_train = self.imputer.fit_transform(X_train)
        X_train = self.scaler.fit_transform(X_train)
        return X_train
    
    def transform(self, X_test):
        """Transform test data using already fitted imputer and scaler"""
        X_test = self.imputer.transform(X_test)
        X_test = self.scaler.transform(X_test)
        return X_test