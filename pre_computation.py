import pandas as pd

from list_creator import get_database_clean, get_mean_similarity, get_all_movies, \
    COLUMNS_SIMILARITY

COLUMNS_USED: set[str] = {"similarity", "validation$r1", "validation$r2"}


def get_light_dataframe(path_to_new_dataframe) -> None:
    """
    Starting from the dataset in PATH_TO_RATINGS, creates a new dataframe containing only
        ['movie1', movie2, 'similarity']. validation$1 and validation$2 become movie1 and movie2, where the suffix
        ".json" is removed.
        Similarity is the mean of the various similarities

    :param path_to_new_dataframe: path where the new dataframe will be saved
    """
    renamed_columns: dict[str, str] = {"validation$r1": "movie1", "validation$r2": "movie2"}

    df_raw: pd.DataFrame = get_database_clean()  # old dataframe
    similarity_column: list[str] = []  # column that contains the mean similarity of each row
    for index, row in df_raw.iterrows():
        similarity_column.append(get_mean_similarity(row))  # append mean similarity to similarity_column
        df_raw.at[index, "validation$r1"] = row["validation$r1"].split(".")[0]  # remove suffix ".json"
        df_raw.at[index, "validation$r2"] = row["validation$r1"].split(".")[0]  # remove suffix ".json"
    df_raw["similarity"] = similarity_column  # add similarity in dataframe

    df_columns_dropped: pd.DataFrame = df_raw.drop(COLUMNS_SIMILARITY, axis=1)  # drop similarity columns

    df_columns_renamed: pd.DataFrame = df_columns_dropped.rename(columns=renamed_columns)  # rename columns

    df_columns_renamed.to_csv(path_to_new_dataframe, index=False)  # save dataframe to path_to_new_dataframe


# def save_top_n_movies_by_popularity(similarities_df: pd.DataFrame, n: int, path_to_save: str) -> None:
#     """
#     The similarity measurements of the top n movies of similarities_df are saved to the path_to_save
#     :param similarities_df: dataframe of movie similarity measurements, it should be the new dataframe with columns
#         ['movie1', 'movie2', 'similarity']
#     :param n: number of movies to select
#     :param path_to_save: path where the similarities will be saved
#     """
#     print(f"saving top {n} movies by popularity")
#
#     movies: pd.DataFrame = get_all_movies(similarities_df)  # get movies of similarities_df
#
#     popularity: dict[int, float] = []
#
#     for index, row in similarities_df:


if __name__ == "__main__":
    print("get_light_dataframe: ")
    #  get_light_dataframe(PATH_TO_NEW_SIMILARITY)  # saves lighter dataframe in PATH_TO_NEW_DATAFRAME
