import os
import pandas as pd
"""
To suppress the warning and opt into the future behavior without making significant changes,
I set the global option at the beginning of your script:
"""
pd.set_option('future.no_silent_downcasting', True)

class DataPreparation:
    """
    A class to handle data preprocessing tasks such as merging datasets,
    handling categorical data, normalization, and feature preparation.
    """

    def __init__(self, main_data_path: str, zip_data_path: str):
        self.main_data_path = main_data_path
        self.zip_data_path = zip_data_path
        self.df = None
        self.zips = None
        self.final_data = None

    def load_data(self):
        self.df = pd.read_csv(self.main_data_path)
        self.zips = pd.read_csv(self.zip_data_path)

    def preprocess(self):
        if "Unnamed: 0" in self.zips.columns:
            self.zips = self.zips.drop(["Unnamed: 0"], axis=1)

        self.df = pd.merge(self.df, self.zips, on="Locality", how="inner")
        self.df["Type_of_Property"] = (
            self.df["Type_of_Property"]
            .replace({"HOUSE": 0, "APARTMENT": 1})
            .astype(int)
        )
        self.df["Region_Code"] = (
            self.df["Region_Code"]
            .replace({1000.0: 1, 2000.0: 2, 3000.0: 3})
            .fillna(0)
            .astype("int64")
        )
        self.df["Terrace"] = (
            self.df["Terrace"].replace({0.0: 0, 1.0: 1}).fillna(0).astype("int64")
        )
        self.df["Garden"] = (
            self.df["Garden"].replace({0.0: 0, 1.0: 1}).fillna(0).astype("int64")
        )
        self.df["Number_of_Facades"] = (
            self.df["Number_of_Facades"]
            .replace({1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4})
            .fillna(0)
            .astype("int64")
        )
        self.df["Swimming_Pool"] = (
            self.df["Swimming_Pool"].replace({0.0: 0, 1.0: 1}).fillna(0).astype("int64")
        )
        self.df["Lift"] = (
            self.df["Lift"].replace({0.0: 0, 1.0: 1}).fillna(0).astype("int64")
        )
        self.df = self.df.drop_duplicates()

    def create_dummies(self):
        state_building = pd.get_dummies(self.df["State_of_the_Building"], dtype=int)
        subtypes_of_property = pd.get_dummies(self.df["Subtype_of_Property"], dtype=int)
        return state_building, subtypes_of_property

    def prepare_features(self):
        state_building, subtypes_of_property = self.create_dummies()
        features_names = [
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
        self.final_data = pd.concat(
            [self.df[features_names], state_building, subtypes_of_property], axis=1
        )

    def save_final_data(self, output_path: str):
        """
        Save the processed dataset to a CSV file after ensuring no duplicates.
        """
        # Remove duplicates from the final dataset
        self.final_data = self.final_data.drop_duplicates()
        self.final_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Get the directory where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths to data files
    main_data_path = os.path.join(script_dir, "../data/immoweb_data_cleaned.csv")
    zip_data_path = os.path.join(script_dir, "../data/zips.csv")
    output_path = os.path.join(script_dir, "../data/immoweb_data_processed.csv")

    # Initialize the DataPreparation class with the correct paths
    data_preparer = DataPreparation(main_data_path=main_data_path, zip_data_path=zip_data_path)

    # Execute data preparation steps
    data_preparer.load_data()
    data_preparer.preprocess()
    data_preparer.prepare_features()
    data_preparer.save_final_data(output_path=output_path)
