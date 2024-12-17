import os
import pandas as pd

# Set global option to suppress the FutureWarning about downcasting behavior
pd.set_option("future.no_silent_downcasting", True)


class DataPreparation:
    """
    A class to handle the complete data preparation process, including:
    - Loading data
    - Preprocessing
    - Creating feature sets
    - Saving the final dataset
    """

    def __init__(self, main_data_path: str, zip_data_path: str):
        """
        Initialize the DataPreparation class with file paths for main and zip datasets.

        Parameters:
        - main_data_path (str): Path to the main dataset CSV file.
        - zip_data_path (str): Path to the zip code dataset CSV file.
        """
        self.main_data_path = main_data_path
        self.zip_data_path = zip_data_path
        self.df = None  # Main dataset
        self.zips = None  # Zip code dataset
        self.final_data = None  # Fully processed dataset

    def load_data(self):
        """
        Load the main and zip datasets into pandas DataFrames.
        """
        self.df = pd.read_csv(self.main_data_path)
        self.zips = pd.read_csv(self.zip_data_path)

    def preprocess(self):
        """
        Perform data preprocessing:
        - Merge datasets on 'Locality'
        - Replace categorical values with numerical equivalents
        - Handle missing values and drop duplicates
        """
        # Remove unnecessary column from the zip dataset if present
        if "Unnamed: 0" in self.zips.columns:
            self.zips = self.zips.drop(["Unnamed: 0"], axis=1)

        # Merge main dataset with zip dataset
        self.df = pd.merge(self.df, self.zips, on="Locality", how="inner")

        # Replace categorical values with numerical equivalents
        replacements = {
            "Type_of_Property": {"HOUSE": 0, "APARTMENT": 1},
            "Region_Code": {1000.0: 1, 2000.0: 2, 3000.0: 3},
            "Terrace": {0.0: 0, 1.0: 1},
            "Garden": {0.0: 0, 1.0: 1},
            "Number_of_Facades": {1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4},
            "Swimming_Pool": {0.0: 0, 1.0: 1},
            "Lift": {0.0: 0, 1.0: 1},
        }

        for column, mapping in replacements.items():
            self.df[column] = self.df[column].replace(mapping).fillna(0).astype("int64")

        # Drop duplicate rows
        self.df = self.df.drop_duplicates()

    def create_dummies(self):
        """
        Create dummy variables for categorical features:
        - 'State_of_the_Building'
        - 'Subtype_of_Property'

        Returns:
        - state_building (DataFrame): Dummy variables for building state.
        - subtypes_of_property (DataFrame): Dummy variables for property subtype.
        """
        state_building = pd.get_dummies(self.df["State_of_the_Building"], dtype=int)
        subtypes_of_property = pd.get_dummies(self.df["Subtype_of_Property"], dtype=int)
        return state_building, subtypes_of_property

    def prepare_features(self):
        """
        Prepare the final dataset with selected features and dummy variables.
        """
        # Generate dummy variables
        state_building, subtypes_of_property = self.create_dummies()

        # Define feature columns to retain
        feature_columns = [
            "Locality",
            "Price",
            "Type_of_Property",
            "Number_of_Rooms",
            "Living_Area",
            "Fully_Equipped_Kitchen",
            "Terrace",
            "Garden",
            "Surface_area_plot_of_land",
            "Number_of_Facades",
            "Swimming_Pool",
            "Lift",
            "Region_Code",
        ]

        # Combine original features with dummy variables
        self.final_data = pd.concat(
            [self.df[feature_columns], state_building, subtypes_of_property], axis=1
        )

    def save_final_data(self, output_path: str):
        """
        Save the fully processed dataset to a CSV file.

        Parameters:
        - output_path (str): Path to save the processed dataset.
        """
        # Remove duplicates and save to file
        self.final_data = self.final_data.drop_duplicates()
        self.final_data.to_csv(output_path, index=False)


class DataPipeline:
    """
    A class to manage the end-to-end data preparation pipeline.
    """

    def __init__(self, main_data_path: str, zip_data_path: str, output_path: str):
        """
        Initialize the DataPipeline class.

        Parameters:
        - main_data_path (str): Path to the main dataset CSV file.
        - zip_data_path (str): Path to the zip code dataset CSV file.
        - output_path (str): Path to save the processed dataset.
        """
        self.data_preparer = DataPreparation(main_data_path, zip_data_path)
        self.output_path = output_path

    def execute_pipeline(self):
        """
        Execute the full data preparation pipeline:
        - Load datasets
        - Preprocess data
        - Prepare features
        - Save the final dataset
        """
        print("Loading datasets...")
        self.data_preparer.load_data()

        print("Preprocessing data...")
        self.data_preparer.preprocess()

        print("Preparing features...")
        self.data_preparer.prepare_features()

        print("Saving final dataset...")
        self.data_preparer.save_final_data(self.output_path)
        print("Data preparation pipeline completed successfully!")


if __name__ == "__main__":
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_data_path = os.path.join(script_dir, "../data/immoweb_data_cleaned.csv")
    zip_data_path = os.path.join(script_dir, "../data/zips.csv")
    output_path = os.path.join(script_dir, "../data/immoweb_data_processed.csv")

    # Run the data preparation pipeline
    pipeline = DataPipeline(main_data_path, zip_data_path, output_path)
    pipeline.execute_pipeline()
