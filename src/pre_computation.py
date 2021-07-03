import os
from shutil import copyfile
from typing import Dict, List, Set, Optional

import pandas as pd
from pandas import DataFrame, Series

from src.similarities_util import get_dataframe_movie_ids_and_similarities, get_mean_similarity, get_movies_from_df, \
    COLUMNS_SIMILARITY, get_movie_dataframe_from_id, get_similarity_dataframe, read_movie_ids_from_csv, \
    PATH_TO_ALL_MOVIES_ID, \
    PATH_TO_SIMILARITY_MPG, PATH_TO_TOP_100_MOVIES_ID, \
    PATH_TO_SIM_100_MPG, PATH_TO_TOP_100_JSON, read_movies_from_csv, \
    get_similar_movies, PATH_TO_TOP_100_SIMILARITIES_MOVIES_ID, convert_tbdb_to_movieId, \
    PATH_TO_SIM_100_MPG_SIMILARITIES, \
    PATH_TO_JSON, PATH_TO_TOP_100_SIMILARITIES_JSON, PATH_TO_SIMILARITY_MP2G, PATH_TO_SIM_100_MP2G, \
    PATH_TO_SIM_100_MP2G_SIMILARITIES, get_genres, \
    get_similarities_with_condition, does_row_contain_only_movies, get_dataframe_of_movie_lists, get_ILS, \
    SimilarityMethod

COLUMNS_MEAN: Set[str] = {"similarity", "validation$r1", "validation$r2"}

# columns of info for the movies
COLUMNS_MOVIES_PLOT_GENRE: List[str] = ["genres", "overview", "similar"]


def write_mean_similarity_dataframe(path_to_new_dataframe: str) -> None:
    """
    Starting from the dataset in PATH_TO_RATINGS, creates a new dataframe containing only
        ['movie1', 'movie2', 'similarity']. validation$1 and validation$2 become movie1 and movie2, where the suffix
        ".json" is removed.
        Similarity is the mean of the various similarities

    :param path_to_new_dataframe: path where the new dataframe will be saved
    """
    renamed_columns: Dict[str, str] = {"validation$r1": "movie1", "validation$r2": "movie2"}

    df_raw: DataFrame = get_dataframe_movie_ids_and_similarities()  # old dataframe
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


def write_mean_similarity_MPG(path_to_new_dataframe: str) -> None:
    """
    Starting from the dataset in PATH_TO_RATINGS, creates a new dataframe containing only
        ['movie1', 'movie2', 'similarity', 'Plot:LDA', 'Genre:Jacc']. validation$1 and validation$2 become movie1 and
        movie2, where the suffix ".json" is removed.
        Similarity is the mean of the various similarities

    :param path_to_new_dataframe: path where the new dataframe will be saved
    """
    print("write_mean_similarity_MPG started...")
    renamed_columns: Dict[str, str] = {"validation$r1": "movie1", "validation$r2": "movie2"}

    print("reading raw dataframe...")
    df_raw: DataFrame = get_dataframe_movie_ids_and_similarities()  # old dataframe
    print("reading raw dataframe done")

    similarity_column: List[str] = []  # column that contains the mean similarity of each row

    print("computing mean similarities...")
    read_elements: int = 0
    for index, row in df_raw.iterrows():
        read_elements += 1
        if read_elements % 10000 == 0:
            print(f"{read_elements} rows done")
        similarity_column.append(get_mean_similarity(row))  # append mean similarity to similarity_column
        df_raw.at[index, "validation$r1"] = row["validation$r1"].split(".")[0]  # remove suffix ".json"
        df_raw.at[index, "validation$r2"] = row["validation$r2"].split(".")[0]  # remove suffix ".json"
    df_raw["similarity"] = similarity_column  # add similarity in dataframe
    print("computing mean similarities done")

    plot_genre_similarities: Set[str] = {'Plot:LDA', 'Genre:Jacc'}
    columns_to_delete: Set[str] = COLUMNS_SIMILARITY.difference(plot_genre_similarities)  # PG are not dropped
    # drop unneeded columns
    df_columns_dropped: DataFrame = df_raw.drop(columns_to_delete, axis=1)

    df_columns_renamed: DataFrame = df_columns_dropped.rename(columns=renamed_columns)  # rename columns

    df_columns_renamed.sort_values(by=['movie1', 'movie2'], inplace=True)  # sort df by movie1 and movie2

    print("writing new dataframe...")
    df_columns_renamed.to_csv(path_to_new_dataframe, index=False)  # save dataframe to path_to_new_dataframe
    print("writing new dataframe done")


def write_mean_similarity_MP2G(path_to_new_dataframe: str) -> None:
    """
    Starting from the dataset in PATH_TO_RATINGS, creates a new dataframe containing only ['movie1', 'movie2',
    'similarity', 'Plot:LDA', 'Plot:cos', 'Genre:Jacc']. validation$1 and validation$2 become movie1 and movie2,
    where the suffix ".json" is removed. Similarity is the mean of the various similarities

    :param path_to_new_dataframe: path where the new dataframe will be saved
    """
    print("write_mean_similarity_MPG started...")
    renamed_columns: Dict[str, str] = {"validation$r1": "movie1", "validation$r2": "movie2"}

    print("reading raw dataframe...")
    df_raw: DataFrame = get_dataframe_movie_ids_and_similarities()  # old dataframe
    print("reading raw dataframe done")

    similarity_column: List[str] = []  # column that contains the mean similarity of each row

    print("computing mean similarities...")
    read_elements: int = 0
    for index, row in df_raw.iterrows():
        read_elements += 1
        if read_elements % 10000 == 0:
            print(f"{read_elements} rows done")
        similarity_column.append(get_mean_similarity(row))  # append mean similarity to similarity_column
        df_raw.at[index, "validation$r1"] = row["validation$r1"].split(".")[0]  # remove suffix ".json"
        df_raw.at[index, "validation$r2"] = row["validation$r2"].split(".")[0]  # remove suffix ".json"
    df_raw["similarity"] = similarity_column  # add similarity in dataframe
    print("computing mean similarities done")

    plot_genre_similarities: Set[str] = {'Plot:LDA', 'Plot:cos', 'Genre:Jacc'}
    columns_to_delete: Set[str] = COLUMNS_SIMILARITY.difference(plot_genre_similarities)  # PG are not dropped
    # drop unneeded columns
    df_columns_dropped: DataFrame = df_raw.drop(columns_to_delete, axis=1)

    df_columns_renamed: DataFrame = df_columns_dropped.rename(columns=renamed_columns)  # rename columns

    df_columns_renamed.sort_values(by=['movie1', 'movie2'], inplace=True)  # sort df by movie1 and movie2

    print("writing new dataframe...")
    df_columns_renamed.to_csv(path_to_new_dataframe, index=False)  # save dataframe to path_to_new_dataframe
    print("writing new dataframe done")


def write_movie_ids_to_csv(movie_ids: List[int], path: str) -> None:
    """
    Writes the movie_ids to a csv specified as path
    :param movie_ids: list of movie ids to write
    :param path: path where to write movie ids
    """
    movie_ids = list(map(int, movie_ids))  # remove decimal .0 from ids
    movie_ids.sort()  # sort movies

    series_ids: Series = Series(movie_ids)
    series_ids.to_csv(path, index=False)  # save series to path_to_movie_ids


def write_all_movies_ids(path_to_movie_ids: str) -> None:
    """
    Writes all movie ids of similarity dataframe to path_to_movie_ids
    :param path_to_movie_ids: path to write ids
    """
    print("write_all_movies_ids starts...")
    print("reading similarities...")
    similarities: DataFrame = get_similarity_dataframe(PATH_TO_SIMILARITY_MPG)
    print("getting movie ids...")
    movie_ids: List[int] = get_movies_from_df(similarities)
    write_movie_ids_to_csv(movie_ids, path_to_movie_ids)


def get_popularity_dict() -> Dict[int, float]:
    """
    Returns a dictionary of movie ids as keys and popularity as value, sorted by reverse popularity
    :return A dictionary of movie ids as keys and popularity as value, sorted by reverse popularity
    """
    movie_ids: List[int] = read_movie_ids_from_csv(PATH_TO_ALL_MOVIES_ID)  # get movies of similarities_df

    popularity_dict: Dict[int, float] = {}  # dictionary of movie id as key, popularity as value

    for movie_id in movie_ids:
        movie: DataFrame = get_movie_dataframe_from_id(movie_id)  # get movie as Dataframe
        popularity: float = movie['tmdb']['popularity']  # get popularity value
        popularity_dict[movie_id] = popularity  # add entry to dictionary

    # sort dict by popularity
    popularity_dict = {k: v for k, v in sorted(popularity_dict.items(), key=lambda item: item[1], reverse=True)}

    return popularity_dict


def copy_movies(movie_ids: List[int], src: str, dst: str):
    """
    Copies the movies whose ids are in movie_ids, from src to dst
    @param movie_ids: ids of movies
    @param src: source of movie files
    @param dst: destination of movie files
    """
    for movie in movie_ids:
        print(movie)
        movie_filename: str = str(movie) + ".json"
        movie_src: str = src + movie_filename
        movie_dst: str = dst + movie_filename
        copyfile(movie_src, movie_dst)


def write_top_n_movies_by_popularity(n: int, path: str) -> None:
    """
    The similarity measurements of the top n movies of similarities_df are saved to path
    :param path: path to write top_n movie_ids
    :param n: number of movies to select
    """
    print(f"saving top {n} movies by popularity")
    # get dictionary of [movie_id, popularity] sorted by popularity
    popularity_dict: Dict[int, float] = get_popularity_dict()
    # get movie_ids sorted by popularity
    movie_ids_sorted_by_popularity: List[int] = list(popularity_dict.keys())

    # get movie_ids of top_n popular movies
    top_n_movies_ids: List[int] = movie_ids_sorted_by_popularity[:n]

    write_movie_ids_to_csv(top_n_movies_ids, path)


def write_similarities_of_movies(path_to_similarities: str, path_to_movies: str, path_to_write: str) -> DataFrame:
    """
    Writes similarities of movies in path_to_movies to path_to_write
    :param path_to_movies: path to movie ids
    :param path_to_write: path to write the similarities
    :param path_to_similarities: path to similarity measurements
    """
    print("write_similarities_of_movies starts...")
    # list of movie ids whose similarity we have to look for. convert to set to remove duplicates
    list_of_movies: List[int] = list(set(read_movie_ids_from_csv(path_to_movies)))

    print("reading similarities dataframe...")
    similarities: DataFrame = get_similarity_dataframe(path_to_similarities)
    print("reading similarities dataframe done")

    print("finding similarities...")
    similarities_of_movies: DataFrame = get_similarities_with_condition(similarities,
                                                                        list_of_movies,
                                                                        does_row_contain_only_movies,
                                                                        print_progress=True)
    print("finding similarities done")

    similarities_of_movies.to_csv(path_to_write, index=False)
    print("write_similarities_of_movies done")

    return similarities_of_movies


def write_top_100_mpg() -> None:
    print("write top 100 mpg starts...")
    write_mean_similarity_MPG(PATH_TO_SIMILARITY_MPG)  # write dataframe of similarities: mean, Plot:LDA, Genre:Jacc
    write_all_movies_ids(PATH_TO_ALL_MOVIES_ID)
    write_top_n_movies_by_popularity(100, PATH_TO_TOP_100_MOVIES_ID)
    write_similarities_of_movies(path_to_movies=PATH_TO_TOP_100_MOVIES_ID, path_to_write=PATH_TO_SIM_100_MPG,
                                 path_to_similarities=PATH_TO_SIMILARITY_MPG)
    # copies json of top n movies
    copy_movies(read_movie_ids_from_csv(PATH_TO_TOP_100_MOVIES_ID),
                PATH_TO_JSON, PATH_TO_TOP_100_JSON)
    print("writing top 100 movies done")


def write_top_100_mp2g() -> None:
    print("write top 100 mp2g starts...")
    # write dataframe of similarities: mean, Plot:LDA, Plot:cos Genre:Jacc
    write_mean_similarity_MP2G(PATH_TO_SIMILARITY_MP2G)
    write_similarities_of_movies(path_to_movies=PATH_TO_TOP_100_MOVIES_ID, path_to_write=PATH_TO_SIM_100_MP2G,
                                 path_to_similarities=PATH_TO_SIMILARITY_MP2G)
    print("writing top 100 movies done")


def write_top_100_mpg_plus_similarities() -> None:
    print("write top 100 mpg plus similarities starts...")
    # list of dataframes of top 100 movies
    top_100_movies: List[DataFrame] = read_movies_from_csv(PATH_TO_TOP_100_MOVIES_ID, PATH_TO_TOP_100_JSON)
    # list of ids of top 100 movies
    top_100_movies_ids: List[int] = read_movie_ids_from_csv(PATH_TO_TOP_100_MOVIES_ID)
    # get ids of recommended movies via JSON
    ids_recommended_movies_from_top100: List[int] = get_similar_movies(top_100_movies)

    # get ids of top 100 recommendations without duplicates via set union
    top_100_similarities_tmdb: List[int] = list(set(ids_recommended_movies_from_top100))
    # convert tmdb id to movieId
    top_100_similarities_movieId: List[int] = convert_tbdb_to_movieId(top_100_similarities_tmdb)

    top_100_plus_similarities: List[int] = \
        list(set(top_100_movies_ids).union(set(top_100_similarities_movieId)))

    # write movie ids
    write_movie_ids_to_csv(top_100_plus_similarities, PATH_TO_TOP_100_SIMILARITIES_MOVIES_ID)

    # write similarities
    write_similarities_of_movies(path_to_movies=PATH_TO_TOP_100_SIMILARITIES_MOVIES_ID,
                                 path_to_write=PATH_TO_SIM_100_MPG_SIMILARITIES,
                                 path_to_similarities=PATH_TO_SIMILARITY_MPG)

    # copies json of movies
    copy_movies(top_100_plus_similarities,
                PATH_TO_JSON, PATH_TO_TOP_100_SIMILARITIES_JSON)

    print("write top 100 mpg plus similarities done")


def write_top_100_mp2g_plus_similarities() -> None:
    print("write top 100 mp2g plus similarities starts...")

    # write similarities
    write_similarities_of_movies(path_to_movies=PATH_TO_TOP_100_SIMILARITIES_MOVIES_ID,
                                 path_to_write=PATH_TO_SIM_100_MP2G_SIMILARITIES,
                                 path_to_similarities=PATH_TO_SIMILARITY_MP2G)

    print("write top 100 mpg plus similarities done")


def mp2g_main():
    """
    Writes the data/top100/similarities_mp2g and data/top100_similarities/similarities_mp2g.
    """
    print("pre computation starts")
    write_top_100_mp2g()
    write_top_100_mp2g_plus_similarities()


def write_movies_info_file(path_to_movies_json: str,
                           path_to_movies_description: str,
                           columns_to_write: List[str]
                           ) -> None:
    """
    Writes a txt file containing the columns in columns_to_write for the movies in the folder path_to_movies_json
    @param path_to_movies_description: path to file of movie description
    @type path_to_movies_description: str
    @param columns_to_write: columns to write for every movie
    @type columns_to_write: Set[str]
    @param path_to_movies_json: path to folder of json for movies
    @type path_to_movies_json: str
    """
    movies_description: str = ""  # string of all movies description

    for movie_path in os.listdir(path_to_movies_json):
        movie_full_path: str = os.path.join(path_to_movies_json, movie_path)
        json_movie_df: DataFrame = pd.read_json(movie_full_path, encoding="UTF-8")
        movie_description: str = ""

        movie_id: int = int(movie_path.split(".")[0])
        movie_title: str = json_movie_df["tmdb"]["title"]

        movie_description += movie_title + ", id = " + str(movie_id) + "\n"

        for column in columns_to_write:
            column_value: str
            if column == "genres":
                column_value = get_genres(json_movie_df["tmdb"][column])
            else:
                column_value = str(json_movie_df["tmdb"][column])
            movie_description += "\t" + column + "\n\t\t" + column_value + "\n"

        movies_description += movie_description  # add the movie description to the overall movies description

    with open(path_to_movies_description, "w") as description_file:
        description_file.write(movies_description)  # write movies description to file


def write_dataframe_ILS(lists_of_ids: List[List[int]],
                        path_to_similarities: str,
                        path_to_write: str,
                        labels: Optional[List[str]]) -> None:
    """
    Writes creates dataframe of ILS for list of list of movies and writes it to path_to_write
    @param labels: Optional labels for movies
    @type labels: str
    @param path_to_similarities: path to similarities for movies
    @type path_to_similarities: str
    @param : dataframe of movies + ILS values
    @type lists_of_ids: List[List[int]]
    @param path_to_write: path to write
    @type path_to_write: str
    """
    similarities_hand_made: DataFrame = get_similarity_dataframe(path_to_similarities)
    df_ILS = get_dataframe_of_movie_lists(lists_of_ids, similarities_hand_made, PATH_TO_JSON)
    if labels is not None:
        df_ILS["label"] = labels
    df_ILS.to_csv(path_to_write)


# def write_list_of_ids_from_list_of_lists(list_of_lists: List[List[int]], path_to_write: str):
#     """
#     Writes the ids in list_of_lists as a Series, in path_to_write
#     @param list_of_lists: List of lists whose ids to write
#     @type list_of_lists: List[List[int]]
#     @param path_to_write: path to write
#     @type path_to_write: str
#     """
#     list_of_ids: List[int] = list(set(matrix_to_list(list_of_lists)))  # convert to list
#     write_movie_ids_to_csv(list_of_ids, path_to_write)


def add_item_to_list_max_ILS(list_to_maximize: List[int],
                             items_to_choose: List[int],
                             similarity_df: DataFrame,
                             similarity_method: SimilarityMethod
                             ) -> Optional[List[int]]:
    """
    Returns a list got by adding one element from items_to_choose to list_to_maximize. The item is chosen in order to
    have ILS maximized.
    @param similarity_method: Method to compute similarity
    @type similarity_method: SimilarityMethod
    @param similarity_df: dataframe of similarity to compute ILS
    @type similarity_df: DataFrame
    @param items_to_choose: list of items to choose in order to maximize ILS of list
    @type items_to_choose: List[int]
    @param list_to_maximize: list to be maximized
    @type list_to_maximize: List[int]
    """
    max_ILS: float = -1  # max ILS value by adding the remaining items
    max_item: Optional[int] = None
    for item in items_to_choose:
        list_maximized_plus_item = list_to_maximize.copy()
        list_maximized_plus_item.append(item)
        ils_i: float = get_ILS(similarity_df, list_maximized_plus_item, similarity_method.value)
        if ils_i is not None and ils_i > max_ILS:  # if ILS is not computable, skip item
            max_ILS = ils_i
            max_item = item
    if max_item is not None:
        list_maximized = list_to_maximize + [max_item]  # add item that keeps ils the highest
        return list_maximized
    else:
        # if there are no similarity for the movies, the first item is added
        list_to_maximize.append(items_to_choose[0])
        return list_to_maximize


if __name__ == "__main__":
    pass
