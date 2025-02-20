import joblib

from config import Config
from data_loader import DataLoader
from model_evaluator import ModelEvaluator
from model_trainer import ModelTrainer
from preprocessor import DataPreprocessor

def main():
    # Initialize components
    config = Config()
    data_loader = DataLoader(config)
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer(config)
    evaluator = ModelEvaluator()

    # Load and prepare data
    wine_df = data_loader.load_wine_data(
        f"{config.DATA_PATH}winequality-red.csv",
        f"{config.DATA_PATH}winequality-white.csv"
    )
    
    weather_df = data_loader.load_weather_data(
        f"{config.DATA_PATH}weather-porto-2024.csv"
    )
    
    # Fix column names for both datasets
    wine_df.columns = wine_df.columns.str.replace(' ', '_')
    weather_df.columns = weather_df.columns.str.replace(' ', '_')
    
    # Merge and preprocess data
    df = data_loader.merge_data(wine_df, weather_df)
    df = preprocessor.preprocess(df)
    
    print(df.columns)
    
    evaluator.plot_hist_show(df)
    
    # Prepare features and split data
    X, y = preprocessor.prepare_features(df)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Scale and impute data
    X_train, X_test = preprocessor.fit_transform(X_train, X_test)
    
    # Train models
    trained_models = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Evaluate models and find the best one
    results, best_model = evaluator.evaluate_models(trained_models, X_test, y_test)

    # Plot results
    evaluator.plot_results(results, X_test, y_test, trained_models[best_model], X, best_model)
    
    # Hyperparameter tuning for RandomForest OR XGBoost because it's the best model
    if best_model == 'random_forest':
        best_model_hyperTuned = trainer.tune_model_RF(X_train, y_train)
    elif best_model == 'xgboost':
        best_model_hyperTuned = trainer.tune_model_XGB(X_train, y_train)
    else:
        best_model_hyperTuned = trained_models[best_model]
        
    # Choose the best model and save it
    joblib.dump(best_model_hyperTuned, f"{config.MODEL_PATH}wine_quality_model.pkl")
    joblib.dump(preprocessor.scaler, f"{config.MODEL_PATH}scaler.pkl")
    joblib.dump(preprocessor.imputer, f"{config.MODEL_PATH}imputer.pkl")

if __name__ == "__main__":
    main()