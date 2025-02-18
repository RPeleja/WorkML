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

from sklearn.model_selection import train_test_split, GridSearchCV
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

    def train_models(self, X_train, y_train, X_test, y_test):
        trained_models = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test,y_test)
            trained_models[name] = model
            print(f"{name} trained. Test Score: {score:.4f}")
        return trained_models
    
    def tune_model_RF(self, X_train, y_train):
        
        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=self.config.RANDOM_STATE)

        # Run Grid Search
        grid_search = GridSearchCV(rf, self.config.param_grid_RF, cv=5, scoring='roc_auc_ovr', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Best parameters & score
        print("Best Params:", grid_search.best_params_)
        print("Best AUC Score:", grid_search.best_score_)
        
        # Train the best model
        best_rf = RandomForestClassifier(
            n_estimators=grid_search.best_params_['n_estimators'],
            max_depth=grid_search.best_params_['max_depth'],
            min_samples_split=grid_search.best_params_['min_samples_split'],
            min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
            max_features=grid_search.best_params_['max_features'],
            bootstrap=grid_search.best_params_['bootstrap'],
            random_state=self.config.RANDOM_STATE
        )

        best_rf.fit(X_train, y_train)
        
        return best_rf
    
    def tune_model_XGB(self, X_train, y_train):
        # Initialize XGBoost classifier
        xgb = XGBClassifier(
            objective='multi:softprob', 
            num_class=3, 
            eval_metric='mlogloss', 
            use_label_encoder=False, 
            random_state=self.config.RANDOM_STATE
        )

        # Run Grid Search with AUC scoring for multiclass (OvR)
        grid_search = GridSearchCV(xgb, self.config.param_grid_XB, cv=5, scoring='roc_auc_ovr', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Best parameters & score
        print("Best Params:", grid_search.best_params_)
        print("Best AUC Score:", grid_search.best_score_)

        # Train the best model
        best_xgb = XGBClassifier(**grid_search.best_params_, objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=self.config.RANDOM_STATE)
        best_xgb.fit(X_train, y_train)

        return best_xgb