Developer by: Rui Peleja & Rui Vieira

WORKML/
├── data/
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   └── weather-porto-2024.csv
├── models/
│   ├── wine_quality_model.pkl
│   ├── scaler.pkl
│   └── imputer.pkl
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   └── main.py
├── Templates/
│   ├── index.html
│   └── result.html
├── app.py
├── requirements.txt
└── README.md


Running the Project

1.Install required packages:
>pip install -r requirements.txt

2.Ensure data files are in the correct location:

    Place wine quality CSV files in the data/ directory
    Place weather data CSV file in the data/ directory

3.Run the main script:
>python src/main.py

Output
The project will:

1.Load and preprocess the data
2.Train multiple models
3.Generate evaluation metrics and plots
4.Save the best performing model and preprocessing objects

Model Performance Interpretation
    Metrics Explained

        Accuracy: Overall prediction accuracy
        Precision: Ratio of true positives to all positive predictions
        Recall: Ratio of true positives to all actual positives
        F1-Score: Harmonic mean of precision and recall
        ROC AUC: Area under the Receiver Operating Characteristic curve

    Understanding the Results

        ROC curves show the trade-off between true positive rate and false positive rate
        Higher AUC indicates better model performance
        Confusion matrices show the distribution of predictions vs actual values

    Extending the Project
        Adding New Models

            1.Add the model to the models dictionary in ModelTrainer
            2.Ensure the model implements fit, predict, and predict_proba methods
            3.Update evaluation metrics if needed

        Adding New Features

            1.Modify the preprocess method in DataPreprocessor
            2.Update the feature selection in prepare_features
            3.Adjust the data loading process if new data sources are added

        Customizing Evaluation

            1.Add new metrics to the evaluate_models method
            2.Create new visualization methods in ModelEvaluator
            3.Modify the results dictionary structure as needed