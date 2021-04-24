import pandas as pd


def save_light_dataframe(path_to_old_dataframe: str, path_to_save: str) -> None:
    """
    Starting from the  dataframe in path_to_old_dataframe, only 3 columns per row remain: 'movie1' containing the id
    of the first movie (without the .json extension), 'movie2' containing the id of the second movie, 'similarity'
    containing the mean of the similarities of the two movies

    :param path_to_old_dataframe: path of old dataframe
    :param path_to_save: path where the new dataframe will be saved
    """


def save_top_n_movies_by_popularity(similarities_df: pd.DataFrame, n: int, path_to_save: str) -> None:
    """
    The similarity measurements of the top n movies of similarities_df are saved to the path_to_save
    :param similarities_df: dataframe of movie similarity measurements
    :param n: number of movies to select
    :param path_to_save: path where the similarities will be saved
    """
    print(f"saving top {n} movies by popularity")




