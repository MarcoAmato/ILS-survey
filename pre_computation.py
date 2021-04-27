import pandas as pd


def save_top_n_movies_by_popularity(similarities_df: pd.DataFrame, n: int, path_to_save: str) -> None:
    """
    The similarity measurements of the top n movies of similarities_df are saved to the path_to_save
    :param similarities_df: dataframe of movie similarity measurements
    :param n: number of movies to select
    :param path_to_save: path where the similarities will be saved
    """
    print(f"saving top {n} movies by popularity")
