from random import sample
import pandas as pd
from os.path import dirname, realpath
from pandas import DataFrame, Series
from typing import List, Set

# folders
# folder where script is/data folder
PATH_TO_DATA_FOLDER = dirname(dirname(realpath(__file__))) + "/data/"
PATH_TO_TOP_100 = PATH_TO_DATA_FOLDER + "top100/"
PATH_TO_TOP_100_SIMILARITIES = PATH_TO_DATA_FOLDER + "top100_similarities/"

# similarity csv
PATH_TO_RAW_SIMILARITY = PATH_TO_DATA_FOLDER + "all_similarities.csv"
PATH_TO_SIMILARITY_MEAN: str = PATH_TO_DATA_FOLDER + "clean_similarity.csv"
PATH_TO_SIMILARITY_MPG: str = PATH_TO_DATA_FOLDER + "similarity_mpg.csv"
PATH_TO_SIM_100_MPG: str = PATH_TO_TOP_100 + "similarities_mpg.csv"  # similarities mpg for top 100 movies
PATH_TO_SIM_100_SIMILARITIES: str = PATH_TO_TOP_100_SIMILARITIES + "similarities_mpg.csv"

# movie ids csv
PATH_TO_ALL_MOVIES_ID: str = PATH_TO_DATA_FOLDER + "all_movies_ids.csv"
PATH_TO_TOP_10_MOVIES_ID: str = PATH_TO_DATA_FOLDER + "top_10_movies_ids.csv"
PATH_TO_TOP_100_MOVIES_ID: str = PATH_TO_TOP_100 + "top_100_movies_ids.csv"
PATH_TO_TOP_100_SIMILARITIES_MOVIES_ID: str = PATH_TO_TOP_100_SIMILARITIES + "movies_ids.csv"

# movie ids conversion
PATH_TO_LINK: str = PATH_TO_DATA_FOLDER + "links.csv"

# path to movie JSON
PATH_TO_JSON = PATH_TO_DATA_FOLDER + "extracted_content_ml-latest/"
PATH_TO_TOP_100_MOVIES_JSON = PATH_TO_TOP_100 + "movies/"
PATH_TO_TOP_100_SIMILARITIES_JSON = PATH_TO_TOP_100_SIMILARITIES + "movies/"

NEW_SIMILARITY_DATAFRAME_COLUMNS = ["movie1", "movie2", "similarity"]
COLUMNS_SIMILARITY = {'Title:LEV', 'Title:JW', 'Title:LCS', 'Title:BI',
                      'Title:LDA', 'Image:EMB', 'Image:BR', 'Image:SH', 'Image:CO',
                      'Image:COL', 'Image:EN', 'Plot:LDA', 'Plot:cos', 'Genre:LDA',
                      'Genre:Jacc', 'Stars:Jacc', 'Directors:Jacc', 'Date:MD', 'Tag:Genome',
                      'SVD'}
MOVIES_LIST_LENGTH = 5


def get_dataframe_movie_ids_and_similarities(num_rows: int = None) -> DataFrame:
    """
    Returns the dataset where the columns are only: the two movies id and the similarity measurements.
    if 'num_rows' is specified, it is the number of similarities we are putting into the dataframe, otherwise we take
    them all
    :param num_rows: number of rows to be read
    :return: the dataframe with movie ids and similarities
    """
    interested_columns = {"validation$r1", "validation$r2"}.union(COLUMNS_SIMILARITY)
    if num_rows is not None and num_rows >= 1:
        return pd.read_csv(PATH_TO_RAW_SIMILARITY, nrows=num_rows, sep="\t", usecols=interested_columns)
    else:
        return pd.read_csv(PATH_TO_RAW_SIMILARITY, sep="\t", usecols=interested_columns)


def get_similarity_dataframe(path: str, num_rows: int = None) -> DataFrame:
    """
    Returns dataframe of similarities, if num_rows is inserted returns first num_rows rows
    :param num_rows: number of rows to be read, if null read all the csv
    :param path: path to dataframe
    """
    similarities: DataFrame
    if num_rows is not None and num_rows >= 1:
        similarities = pd.read_csv(path, nrows=num_rows)
    else:
        similarities = pd.read_csv(path)
    similarities.movie1 = similarities.movie1.astype(int)  # remove .0 suffix
    similarities.movie2 = similarities.movie2.astype(int)  # remove .0 suffix
    return similarities


def get_similarities_of_movies(similarities: DataFrame, list_of_movies: List[int]) -> DataFrame:
    """
    Returns a dataframe containing all the similarities between the movies in list_of_movies
    :param similarities: Dataframe containing all the similarities
    :param list_of_movies: list of movie ids whose similarities we want to retrieve
    """
    # create dataframe for similarities of list_of_movies
    movies_similarities = DataFrame(columns=NEW_SIMILARITY_DATAFRAME_COLUMNS)

    rows_read: int = 0
    for index, similarity_row in similarities.iterrows():
        if rows_read % 100000 == 0:
            print(f"{rows_read} rows read")
        rows_read += 1
        if similarity_row.movie1 in list_of_movies and similarity_row.movie2 in list_of_movies:
            # similarity of 2 movies in list_of_movies
            movies_similarities = movies_similarities.append(similarity_row)

    movies_similarities.movie1 = movies_similarities.movie1.astype(int)  # movie1 treated as int
    movies_similarities.movie2 = movies_similarities.movie2.astype(int)  # movie2 treated as int
    print(movies_similarities)
    return movies_similarities


def get_movies_from_df(df_similarities: DataFrame) -> List[int]:
    """
    Returns list of all the movies in 'df'
    :param df_similarities: dataframe of similarities
    :return: list of movies ids
    """
    movies: List[int] = []
    for index, row in df_similarities.iterrows():
        if row.movie1 not in movies:
            movies.append(row.movie1)
        elif row.movie2 not in movies:
            movies.append(row.movie2)
    return movies


def read_movie_ids_from_csv(path: str) -> List[int]:
    """
    Returns ids of all movies read from path
    :param path: path were movie csv is
    :return: corresponding series of movie ids
    """

    df: DataFrame = pd.read_csv(path, index_col=False, header=0)
    series: Series = df.iloc[:, 0]
    return series.tolist()


def read_movies_from_csv(path: str) -> List[DataFrame]:
    """
    Returns dataframe movies whose ids are in path
    :param path: path where movie ids are
    :return: Dataframe of movies
    """
    movie_ids: List[int] = read_movie_ids_from_csv(path)
    return get_movies_by_id(movie_ids)


def get_similar_movies(movies_dataframes: List[DataFrame]) -> List[int]:
    """
    Returns list of movie ids who are inserted in the column "recommendations" for the movies
    passed as movies_ids
    @param movies_dataframes: List of movie Dataframe to check for recommendations
    """
    similarities_for_list: Set[int] = set()
    for movie_df in movies_dataframes:
        similarities_for_movie: Set[int] = movie_df['tmdb']['similar']
        similarities_for_list = similarities_for_list.union(similarities_for_movie)
    list_of_similarities: List[int] = list(similarities_for_list)
    return list_of_similarities


def convert_tbdb_to_movieId(movie_ids_tmdb: List[int]) -> List[int]:
    """
    Returns the ids in movie_ids_tmdb to the list of the same movies with corresponding in movieId.
    @param movie_ids_tmdb: List of movie ids to tmdb format.
    """
    links_csv: DataFrame = pd.read_csv(PATH_TO_LINK)
    rows_of_movies: DataFrame = links_csv.loc[links_csv['tmdbId'].isin(movie_ids_tmdb)]
    print(rows_of_movies)
    movie_ids_movieId: List[int] = rows_of_movies['movieId']
    return movie_ids_movieId


def get_movies_by_id(list_of_movies: List[int]) -> List[DataFrame]:
    """
    Return dataframe of movies whose ids were passed by list_of_movies
    :param list_of_movies: list of movie ids
    :returns list of dataframe of movies
    """
    movies: List[DataFrame] = []
    for movie_id in list_of_movies:
        movies.append(get_movie_from_top100(movie_id))
    return movies


def get_movie(movie_id: int) -> DataFrame:
    """
    Return dataframe of movie
    :param movie_id: id of movie
    :return: dataframe of movie
    """
    path: str = PATH_TO_JSON + str(movie_id) + ".json"
    return pd.read_json(path)


def get_movie_from_top100(movie_id: int) -> DataFrame:
    """
    Return dataframe of movie reading the path
    :param movie_id: id of movie
    :return: dataframe of movie
    """
    path: str = PATH_TO_TOP_100_MOVIES_JSON + str(movie_id) + ".json"
    return pd.read_json(path)


def get_name_of_movie(movie: DataFrame) -> str:
    return movie["tmdb"]["title"]


def get_mean_similarity(similarity_row: pd.Series):
    similarity_values = similarity_row[COLUMNS_SIMILARITY]
    return sum(similarity_values) / len(similarity_values)


def print_movie_with_info(movie_id: int, columns_to_print: List[str]) -> None:
    """
    Prints the columns_to_print columns of movie_id
    @param movie_id: Id of movie to print
    @param columns_to_print: List of columns to print
    """
    movie_df: DataFrame = get_movie(movie_id)
    print("++++++++++")
    print("Movie: " + get_name_of_movie(movie_df))
    for column in columns_to_print:
        print("\t" + column)
        print(movie_df['tmdb'][column])
    print("----------")


def get_ILS(similarity_measures: pd.DataFrame, list_of_movies: List[int], method: str) -> float:
    """
    Returns ILS value for the list_of_movies using the similarity_measures
    :param similarity_measures: dataframe of similarity measurements
    :param list_of_movies: list of movies ids
    :param method: method to compute ILS
    :return: ILS value for list_of_movies using the similarities in similarity_measures
    """
    # get similarity Dataframe for the movies in list_of_movies
    similarities_of_movies: DataFrame = get_similarities_of_movies(similarity_measures, list_of_movies)
    ILS: float = 0
    if method == "mean":
        ILS = similarities_of_movies['similarity'].sum()
    elif method == "plot":
        for movie in list_of_movies:
            print_movie_with_info(movie, list("overview"))
        ILS = similarities_of_movies['Plot:LDA'].sum()
    elif method == "genre":
        for movie in list_of_movies:
            print_movie_with_info(movie, list("genres"))
        ILS = similarities_of_movies['Genre:Jacc'].sum()
    elif method == "plot-genre":
        for movie in list_of_movies:
            print_movie_with_info(movie, ["genres", "plot"])
        # mean of plot and genre
        ILS = (similarities_of_movies['Plot:LDA'].sum() + similarities_of_movies['Genre:Jacc'].sum()) / 2

    # we normalize using the number of similarities
    ILS_normalized: float = ILS / similarities_of_movies.shape[0]
    return ILS_normalized


def get_similarity(similarity_df: DataFrame, movie1: int, movie2: int) -> float:
    """
    Returns the similarity of movie1 and movie2 based on similarity_df
    :param similarity_df: dataframe of similarities
    :param movie1: id of movie1
    :param movie2: id of movie2
    :return: similarity of movie1 and movie2 based on similarity_df
    """
    for index, similarity_row in similarity_df.iterrows():
        if (similarity_row.movie1 == movie1 and similarity_row.movie2 == movie2) \
                or (similarity_row.movie2 == movie1 and similarity_row.movie1 == movie2):  # the 2 movies are in the row
            return similarity_row.similarity  # get similarity of the row
    print("error")
    return 0


def test_top_10_movies():
    print("test_top_10_movies")
    similarities_df: DataFrame = get_similarity_dataframe(PATH_TO_SIMILARITY_MPG)
    top_10_movie_ids: List[int] = read_movie_ids_from_csv(PATH_TO_TOP_10_MOVIES_ID)

    # similarities_top_10 = get_similarities_of_movies(similarities_df, top_10_movie_ids)

    # get random list of MOVIES_LIST_LENGTH movies
    test_list_of_movies: List[int] = sample(top_10_movie_ids, MOVIES_LIST_LENGTH)

    ILS_m: float = get_ILS(similarities_df, test_list_of_movies, "mean")
    ILS_p: float = get_ILS(similarities_df, test_list_of_movies, "plot")
    ILS_g: float = get_ILS(similarities_df, test_list_of_movies, "genre")

    print(test_list_of_movies)
    print(ILS_m)
    print(ILS_p)
    print(ILS_g)


def print_names_of_movies(movie_ids: List[int]) -> None:
    """
    Prints names of movies whose ids are in movie_ids
    @param movie_ids: ids of movies
    """
    movies_dataframes: List[DataFrame] = get_movies_by_id(movie_ids)
    for movie in movies_dataframes:
        movie_name: str = get_name_of_movie(movie)
        print("\t" + movie_name)


def print_ils_top_100_MPG() -> None:
    """
    Finds MOVIES_LIST_LENGTH movies from the top 100 popularity movies and computes mean with:
        mean similarity, Plot and Genre
    """
    print("test_top_100_movies")
    # MPG stands for Mean (similarity), Plot, Genre
    similarities_df: DataFrame = get_similarity_dataframe(PATH_TO_SIM_100_MPG)
    top_100_movie_ids: List[int] = read_movie_ids_from_csv(PATH_TO_TOP_100_MOVIES_ID)
    sample_list_of_movies: List[int] = sample(top_100_movie_ids, MOVIES_LIST_LENGTH)

    ILS_m: float = get_ILS(similarities_df, sample_list_of_movies, "mean")
    ILS_p: float = get_ILS(similarities_df, sample_list_of_movies, "plot")
    ILS_g: float = get_ILS(similarities_df, sample_list_of_movies, "genre")
    ILS_pg: float = get_ILS(similarities_df, sample_list_of_movies, "plot-genre")

    print("movies: ")
    print_names_of_movies(sample_list_of_movies)
    # print(sample_list_of_movies)
    print("ILD using mean of similarities: ")
    print(ILS_m)
    print("ILD using mean of Plot:LDA: ")
    print(ILS_p)
    print("ILD using mean of Genre:JACC: ")
    print(ILS_g)
    print("ILD using mean of Plot and Genre: ")
    print(ILS_pg)
