# Movie Audience Rating Prediction Model

## Overview
This project implements a machine learning model to predict movie audience ratings using the Rotten Tomatoes dataset. The model analyzes various movie features including genre, directors, runtime, and release timing to predict audience ratings.

## Dataset Features
- Movie rating (PG, PG-13, R, etc.)
- Genre
- Directors
- Writers
- Studio name
- Tomatometer status
- Tomatometer rating and count
- Runtime in minutes
- Theater release date
- Streaming release date
- Audience rating (target variable)

## Data Preprocessing

### Data Cleaning
- Removed null values in audience_rating column
- Dropped high cardinality columns:
  - movie_title
  - movie_info
  - critics_consensus
  - cast
  - in_theaters_date
  - on_streaming_date
- Handled missing values in directors column

### Feature Engineering

#### Director Categories
Categorized directors based on number of movies directed:
- 5+ Movies
- 4 Movies
- 3 Movies
- 2 Movies
- 1 Movie

#### Runtime Categories
- Initial categorization: 30-minute intervals (1-30, 31-60, etc.)
- Refined categorization:
  - 1-60 minutes
  - 61-120 minutes
  - 121-180 minutes
  - 181-300+ minutes

#### Release Month Features
- Extracted theater release month
- Extracted streaming release month

## Pipeline Architecture

### Custom Transformers
- **ListToStringTransformer**
  - Converts list features to string format
- **FrequencyEncoder**
  - Encodes categorical variables based on their frequency

### Specialized Pipelines
- **Rating Pipeline**
  - Imputation with Simple Imputer
  - One-hot encoding
- **Genre Pipeline**
  - List to string Conversion using Transformer
  - Imputation with Simple imputer
  - One-hot encoding
- **directors Pipeline**
  - Imputation with Simple imputer
  - Frquency encoding
- **writers Pipeline**
  - Imputation with Simple imputer
  - Frquency encoding
- **Studio Pipeline**
  - Imputation with Simple imputer
  - Frquency encoding
- **Tomatometer_status Pipeline**
  - Imputation with Simple imputer
  - Ordinal encoding
- **Numerical Pipeline**
  - Median imputation with Simple imputer
  - Standard scaling

## Models Implemented

### 1. Random Forest Regressor
Parameters:
- n_estimators: 300
- min_samples_split: 3
- min_samples_leaf: 5
- max_features: sqrt
- bootstrap: True
- oob_score: True

### 2. Gradient Boosting Regressor
Parameters:
- n_estimators: 100
- learning_rate: 0.05
- max_depth: 3
- min_samples_split: 10
- min_samples_leaf: 10
- subsample: 0.9
- loss: huber

## Model Evaluation
- Used R-squared score and Mean Squared Error
- Implemented validation set for model performance assessment
- Data split:
  - 70% Training
  - 15% Validation
  - 15% Test

## Future Improvements

### Dataset Enhancement
- Current dataset has only 98 unique audience rating values
- Need more data for better model accuracy

### Model Optimization
- Implement hyperparameter tuning using GridSearch/RandomSearch
- Analyze feature correlation with target variable
- Extract feature importance scores from RandomForest

## Requirements
numpy
pandas
scikit-learn
seaborn
matplotlib
google.colab

## Contact
- Email: chandima.seanrathna97@gmail.com
- Phone: +94787580595

## Notes
This project demonstrates a complete machine learning pipeline from data preprocessing to model evaluation. The implementation focuses on handling different types of features (categorical, numerical, list) and comparing multiple regression models
