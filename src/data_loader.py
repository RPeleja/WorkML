"""
    Handles loading and merging of wine and weather datasets.
    
    Methods:
        load_wine_data(red_wine_path, white_wine_path): 
            Loads and combines red and white wine datasets
            Returns: pandas.DataFrame with combined wine data
            
        load_weather_data(weather_path):
            Loads weather data and converts timestamps
            Returns: pandas.DataFrame with weather data
            
        merge_data(wine_df, weather_df):
            Merges wine and weather data based on timestamps
            Returns: pandas.DataFrame with merged data
    
    Usage:
        loader = DataLoader(config)
        wine_df = loader.load_wine_data('red.csv', 'white.csv')
        weather_df = loader.load_weather_data('weather.csv')
        merged_df = loader.merge_data(wine_df, weather_df)
"""

import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_wine_data(self, red_wine_path, white_wine_path):
        df_red = pd.read_csv(red_wine_path)
        df_red['type'] = '0'
        df_white = pd.read_csv(white_wine_path)
        df_white['type'] = '1'
        return pd.concat([df_red, df_white], ignore_index=True)

    def load_weather_data(self, weather_path):
        weather_df = pd.read_csv(weather_path, delimiter=';')
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], format='%d/%m/%Y %H:%M')
        return weather_df

    def merge_data(self, wine_df, weather_df):
        weather_filtered = weather_df[
            (weather_df['timestamp'].dt.month.isin(self.config.WEATHER_MONTHS))
        ]

        # Generate random dates for wine data
        min_date = weather_filtered['timestamp'].min()
        max_date = weather_filtered['timestamp'].max()
        wine_df['timestamp'] = np.random.choice(
            pd.date_range(min_date, max_date), 
            size=len(wine_df)
        )

        merged_df = pd.merge_asof(
            wine_df.sort_values('timestamp'),
            weather_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        # Extracting features from timestamp
        merged_df['hour'] = merged_df['timestamp'].dt.hour
        merged_df['day'] = merged_df['timestamp'].dt.day
        merged_df['month'] = merged_df['timestamp'].dt.month
        merged_df['weekday'] = merged_df['timestamp'].dt.weekday
        
        # Create a feature for time of day (morning, afternoon, night)
        merged_df['time_of_day'] = merged_df['hour'].apply(lambda x: 'morning' if 6 <= x < 12 
                                                                    else 'afternoon' if 12 <= x < 18 
                                                                    else 'night')

        # Create seasonal feature based on month
        merged_df['season'] = merged_df['month'].apply(lambda x: 'winter' if x in [12, 1, 2] 
                                                                else 'spring' if x in [3, 4, 5] 
                                                                else 'summer' if x in [6, 7, 8] 
                                                                else 'autumn')

        # Convert categorical features into numerical (one-hot encoding)
        merged_df = pd.get_dummies(merged_df, columns=['time_of_day', 'season'], drop_first=True)

        # One-Hot Encode categorical columns (e.g., entity_id, name)
        merged_df = pd.get_dummies(merged_df, columns=['entity_id', 'name'], drop_first=True)
        
        # Apply Log Transformation to Skewed Features
        skewed_features = ["residual_sugar", "free_sulfur_dioxide", "total_sulfur_dioxide"]
        for feature in skewed_features:
            merged_df[feature] = np.log1p(merged_df[feature])
        
        return merged_df