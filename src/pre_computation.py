from shutil import copyfile
from typing import Dict, List, Set

from pandas import DataFrame, Series

from src.similarities_util import get_dataframe_movie_ids_and_similarities, get_mean_similarity, get_movies_from_df, \
    COLUMNS_SIMILARITY, get_movie_dataframe_from_id, get_similarity_dataframe, read_movie_ids_from_csv, \
    PATH_TO_ALL_MOVIES_ID, \
    PATH_TO_SIMILARITY_MPG, PATH_TO_TOP_100_MOVIES_ID, \
    get_similarities_of_movies, PATH_TO_SIM_100_MPG, PATH_TO_TOP_100_MOVIES_JSON, read_movies_from_csv, \
    get_similar_movies, PATH_TO_TOP_100_SIMILARITIES_MOVIES_ID, convert_tbdb_to_movieId, PATH_TO_SIM_100_SIMILARITIES, \
    PATH_TO_JSON, PATH_TO_TOP_100_SIMILARITIES_JSON

COLUMNS_MEAN: Set[str] = {"similarity", "validation$r1", "validation$r2"}


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


def write_similarities_of_movies(path_to_similarities: str, path_to_movies: str, path_to_write: str) -> None:
    """
    Writes similarities of movies in path_to_movies to path_to_write
    :param path_to_movies: path to movie ids
    :param path_to_write: path to write the similarities
    :param path_to_similarities: path to similarity measurements
    """
    print("write_similarities_of_movies starts...")
    # list of movie ids whose similarity we have to look for
    top_n_movies: List[int] = read_movie_ids_from_csv(path_to_movies)

    print("reading similarities dataframe...")
    similarities: DataFrame = get_similarity_dataframe(path_to_similarities)
    print("reading similarities dataframe done")
    # pd.read_csv(path_to_similarities)

    print("finding similarities...")
    similarities_of_movies: DataFrame = get_similarities_of_movies(similarities, top_n_movies)
    print("finding similarities done")

    similarities_of_movies.to_csv(path_to_write, index=False)
    print("write_similarities_of_movies done")


def write_top_100_mpg() -> None:
    write_mean_similarity_MPG(PATH_TO_SIMILARITY_MPG)  # write dataframe of similarities: mean, Plot:LDA, Genre:Jacc
    write_all_movies_ids(PATH_TO_ALL_MOVIES_ID)
    write_top_n_movies_by_popularity(100, PATH_TO_TOP_100_MOVIES_ID)
    write_similarities_of_movies(path_to_movies=PATH_TO_TOP_100_MOVIES_ID, path_to_write=PATH_TO_SIM_100_MPG,
                                 path_to_similarities=PATH_TO_SIMILARITY_MPG)
    # copies json of top n movies
    copy_movies(read_movie_ids_from_csv(PATH_TO_TOP_100_MOVIES_ID),
                PATH_TO_JSON, PATH_TO_TOP_100_MOVIES_JSON)


def write_top_100_mpg_plus_similarities() -> None:
    print("top 100 mpg plus similarities starts...")
    # list of dataframes of top 100 movies
    top_100_movies: List[DataFrame] = read_movies_from_csv(PATH_TO_TOP_100_MOVIES_ID)
    # list of ids of top 100 movies
    top_100_movies_ids: List[int] = read_movie_ids_from_csv(PATH_TO_TOP_100_MOVIES_ID)
    # get ids of recommended movies via JSON
    ids_recommended_movies_from_top100: List[int] = get_similar_movies(top_100_movies)

    # get ids of top 100 recommendations without duplicates via set union
    top_100_similarities_tmdb: List[int] = list(set(ids_recommended_movies_from_top100))
    # convert tmdb id to movieId
    top_100_plus_similarities_movieId: List[int] = convert_tbdb_to_movieId(top_100_similarities_tmdb)

    top_100_plus_similarities: List[int] = \
        list(set(top_100_movies_ids).union(set(top_100_plus_similarities_movieId)))

    # write movie ids
    write_movie_ids_to_csv(top_100_plus_similarities, PATH_TO_TOP_100_SIMILARITIES_MOVIES_ID)

    # write similarities
    write_similarities_of_movies(path_to_movies=PATH_TO_TOP_100_SIMILARITIES_MOVIES_ID,
                                 path_to_write=PATH_TO_SIM_100_SIMILARITIES,
                                 path_to_similarities=PATH_TO_SIMILARITY_MPG)

    # copies json of movies
    copy_movies(top_100_plus_similarities_movieId,
                PATH_TO_JSON, PATH_TO_TOP_100_SIMILARITIES_JSON)

    print("top 100 mpg plus similarities done")


if __name__ == "__main__":
    print("pre computation starts")
    write_top_100_mpg_plus_similarities()
