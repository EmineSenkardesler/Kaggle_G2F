# Kaggle G2F Competition - Python Models

This directory contains Python implementations of machine learning models for the Kaggle G2F (Genomes to Fields) competition. The goal is to predict the binary target variable `Win_YN` based on various environmental and genetic features.

## Files Description

*   **`rf_model.ipynb`**:
    *   Implements a **Random Forest Classifier**.
    *   Performs data preprocessing including One-Hot Encoding and handling categorical variables.
    *   Splits data into training and testing sets.
    *   Evaluates the model using Confusion Matrix, Accuracy, and ROC/AUC.
    *   Generates predictions for the Kaggle competition dataset (`kaggle_prediction.csv`) and saves them to `kaggle_prediction_output_rf_notebook.csv`.

*   **`g2f_models.ipynb`**:
    *   Explores multiple classification models:
        *   **Logistic Regression (GLM)**
        *   **Multi-Layer Perceptron (MLP) Classifier (NNET)**
        *   **Gradient Boosting Classifier (GBM)**
    *   Includes data preprocessing steps like Target Encoding, One-Hot Encoding, Correlation Filtering, and Scaling.
    *   Splits the dataset into 80% training and 20% testing.
    *   Evaluates and compares models based on Accuracy and AUC.

*   **`G2F_data.csv`**: The main dataset used for training and testing.
*   **`kaggle_prediction.csv`**: The test dataset for Kaggle competition.
*   **`kaggle_prediction_output.csv`**: Output file containing predictions.
*   **`kaggle_prediction_output_rf_notebook.csv`**: Output file containing predictions specifically from the Random Forest notebook.

## Dependencies

To run these notebooks, you need the following Python libraries:

*   pandas
*   numpy
*   matplotlib
*   scikit-learn (sklearn)

## Usage

1.  Ensure all dependencies are installed.
2.  Open the notebooks (`.ipynb` files) in Jupyter Notebook or JupyterLab.
3.  Run the cells sequentially to load data, train models, and view results.
