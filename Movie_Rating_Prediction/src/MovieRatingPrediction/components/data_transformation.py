from src.MovieRatingPrediction.logger import logging

class DataTransform:
    def __init__(self, data):
        self.data = data
        self.transformed_data = None

    def data_transform(self):
        # Dropping rows with missing values
        self.data.dropna(subset=["Name", "Year", "Duration", "Genre", "Rating", "Votes", "Director"], inplace=True)

        # Transforming 'Duration', 'Year', and 'Votes' columns
        self.data['Duration'] = self.data['Duration'].str.replace('min', '').astype(int)
        self.data['Year'] = self.data['Year'].str.strip('()').astype(int)
        self.data['Votes'] = self.data['Votes'].str.replace(",", "").astype(int)

        # Dropping unnecessary columns
        self.data.drop(["Name", "Director", "Genre", "Actor 1", "Actor 2", "Actor 3"], inplace=True, axis=1)

        # Store the transformed data
        self.transformed_data = self.data.copy()

        # Logging information about the transformation
        if self.transformed_data is not None:
            logging.info("Data is successfully transformed.")
            return self.transformed_data
        else:
            logging.info("Data has not been transformed yet. Call data_transform method first.")