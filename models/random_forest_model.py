import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

class ModelEvaluator:
    """
    A class to handle the evaluation of a machine learning model's performance.
    It calculates various metrics such as R², MSE, RMSE, and MAE.
    """

    @staticmethod
    def evaluate_model(y_true, y_pred, dataset_type):
        """
        Evaluate the model's performance on a given dataset.

        Parameters:
        - y_true: Actual target values.
        - y_pred: Predicted target values.
        - dataset_type: Type of dataset (e.g., 'Training' or 'Testing').
        """
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        print(f"{dataset_type} Set Performance:")
        print(f"R² Score: {r2}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print()

class RandomForestModel:
    """
    A class for training and evaluating a Random Forest model for regression.
    """

    def __init__(self, data_path):
        """
        Initialize the RandomForestModel with dataset path and load the data.

        Parameters:
        - data_path: Path to the dataset CSV file.
        """
        self.df = pd.read_csv(data_path)
        self.rf_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )

    def prepare_data(self):
        """
        Prepare features and target variables (X and y).

        Returns:
        - X: Feature data.
        - y: Target data.
        """
        X = self.df[
            [
                "Locality",
                "Type_of_Property",
                "Number_of_Rooms",
                "Living_Area",
                "Fully_Equipped_Kitchen",
            ]
        ]
        y = self.df["Price"]
        return X, y

    def train_model(self, X_train, y_train):
        """
        Train the Random Forest model using the provided training data.

        Parameters:
        - X_train: Feature data for training.
        - y_train: Target data for training.
        """
        self.rf_regressor.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: Feature data for prediction.

        Returns:
        - Predicted values.
        """
        return self.rf_regressor.predict(X)

    def save_model(self, model_path):
        """
        Save the trained model to a file in .joblib format.

        Parameters:
        - model_path: Path to save the .joblib file.
        """
        joblib.dump(self.rf_regressor, model_path)

    def cross_validate(self, X, y):
        """
        Perform cross-validation on the model to evaluate its performance.

        Parameters:
        - X: Feature data.
        - y: Target data.
        """
        cv_scores = cross_val_score(
            self.rf_regressor, X, y, cv=10, scoring="r2", n_jobs=-1
        )
        print("Cross-Validated R² Scores:", cv_scores)
        print("Mean R² Score from Cross-Validation:", np.mean(cv_scores))

class RandomForestPipeline:
    """
    A pipeline to manage the entire Random Forest regression workflow.
    """

    def __init__(self, data_path, model_path):
        """
        Initialize the pipeline with data and model paths.

        Parameters:
        - data_path: Path to the dataset.
        - model_path: Path to save the trained model.
        """
        self.data_path = data_path
        self.model_path = model_path
        self.rf_model = RandomForestModel(data_path)

    def run(self):
        """
        Execute the pipeline: prepare data, train model, evaluate, save model, and cross-validate.
        """
        # Prepare data
        X, y = self.rf_model.prepare_data()

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.22, random_state=42
        )

        # Train the model
        self.rf_model.train_model(X_train, y_train)

        # Make predictions
        pred_train = self.rf_model.predict(X_train)
        pred_test = self.rf_model.predict(X_test)

        # Evaluate the model
        evaluator = ModelEvaluator()
        evaluator.evaluate_model(y_train, pred_train, "Training")
        evaluator.evaluate_model(y_test, pred_test, "Testing")

        # Save the trained model
        self.rf_model.save_model(self.model_path)
        print("Model saved successfully!")

        # Cross-validation
        self.rf_model.cross_validate(X, y)

def main():
    """
    Main function to execute the Random Forest regression pipeline.
    """
    data_path = "./data/immoweb_data_processed.csv"
    model_path = "./models/random_forest_model.joblib"

    pipeline = RandomForestPipeline(data_path, model_path)
    pipeline.run()

if __name__ == "__main__":
    main()