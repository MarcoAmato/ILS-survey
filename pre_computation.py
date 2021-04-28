import pandas as pd
from pandas import DataFrame, Series
from typing import Dict, Set, List

from list_creator import get_database_clean, get_mean_similarity, get_movies_from_df, \
    COLUMNS_SIMILARITY, PATH_TO_NEW_SIMILARITY, get_movie, get_light_dataframe, PATH_TO_LITTLE_SIMILARITY, \
    PATH_TO_ALL_MOVIES_ID, read_movie_ids_from_csv

COLUMNS_USED: set[str] = {"similarity", "validation$r1", "validation$r2"}


def write_light_dataframe(path_to_new_dataframe: str) -> None:
    """
    Starting from the dataset in PATH_TO_RATINGS, creates a new dataframe containing only
        ['movie1', movie2, 'similarity']. validation$1 and validation$2 become movie1 and movie2, where the suffix
        ".json" is removed.
        Similarity is the mean of the various similarities

    :param path_to_new_dataframe: path where the new dataframe will be saved
    """
    renamed_columns: Dict[str, str] = {"validation$r1": "movie1", "validation$r2": "movie2"}

    df_raw: DataFrame = get_database_clean()  # old dataframe
    similarity_column: List[str] = []  # column that contains the mean similarity of each row
    for index, row in df_raw.iterrows():
        similarity_column.append(get_mean_similarity(row))  # append mean similarity to similarity_column
        df_raw.at[index, "validation$r1"] = row["validation$r1"].split(".")[0]  # remove suffix ".json"
        df_raw.at[index, "validation$r2"] = row["validation$r2"].split(".")[0]  # remove suffix ".json"
    df_raw["similarity"] = similarity_column  # add similarity in dataframe

    df_columns_dropped: DataFrame = df_raw.drop(COLUMNS_SIMILARITY, axis=1)  # drop similarity columns

    df_columns_renamed: DataFrame = df_columns_dropped.rename(columns=renamed_columns)  # rename columns

    df_columns_renamed.sort_values(by=['movie1', 'movie2'], inplace=True)  # sort df by movie1 and movie2

    df_columns_renamed.to_csv(path_to_new_dataframe, index=False)  # save dataframe to path_to_new_dataframe


def write_all_movies_ids(path_to_movie_ids: str) -> None:
    """
    Writes all movie ids of similarity dataframe to path_to_movie_ids
    :param path_to_movie_ids: path to write ids
    """
    print("write_all_movies_ids starts...")
    print("reading similarities...")
    similarities: DataFrame = get_light_dataframe()
    print("getting movie ids...")
    movie_ids: List[int] = get_movies_from_df(similarities)
    movie_ids = list(map(int, movie_ids))
    movie_ids.sort()  # sort movies

    dataframe_ids: Series = Series(movie_ids)
    print(dataframe_ids)
    dataframe_ids.to_csv(path_to_movie_ids, index=False)  # save series to path_to_movie_ids


def get_popularity_dict() -> Dict[int, float]:
    """
    Returns a dictionary of movie ids as keys and popularity as value, sorted by reverse popularity
    :return A dictionary of movie ids as keys and popularity as value, sorted by reverse popularity
    """
    movie_ids: Series = read_movie_ids_from_csv()  # get movies of similarities_df

    popularity_dict: Dict[int, float] = {}  # dictionary of movie id as key, popularity as value

    for index, movie_id in movie_ids.items():
        movie: DataFrame = get_movie(movie_id)  # get movie as Dataframe
        popularity: float = movie['tmdb']['popularity']  # get popularity value
        popularity_dict[movie_id] = popularity  # add entry to dictionary

    # sort dict by popularity
    popularity_dict = {k: v for k, v in sorted(popularity_dict.items(), key=lambda item: item[1], reverse=True)}

    return popularity_dict


def save_top_n_movies_by_popularity(n: int) -> None:
    """
    The similarity measurements of the top n movies of similarities_df are saved to the PATH_TO_LITTLE_SIMILARITY
    :param n: number of movies to select
    """
    print(f"saving top {n} movies by popularity")
    # get dictionary of [movie_id, popularity] sorted by popularity
    popularity_dict: Dict[int, float] = get_popularity_dict()
    # get movie_ids sorted by popularity
    movie_ids_sorted_by_popularity: List[int] = list(popularity_dict.keys())
    # get movie_ids of top_n popular movies
    top_n_movies_ids: List[int] = movie_ids_sorted_by_popularity[:n]

    for movie_id in top_n_movies_ids:
        print(movie_id)


if __name__ == "__main__":
    save_top_n_movies_by_popularity(10)
