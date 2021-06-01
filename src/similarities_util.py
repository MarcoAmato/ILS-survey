import random
from random import sample
import pandas as pd
from os.path import dirname, realpath
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from typing import List, Set, Optional, Dict

# folders
# folder where script is/data folder
PATH_TO_DATA_FOLDER = dirname(dirname(realpath(__file__))) + "/data/"
PATH_TO_TOP_100 = PATH_TO_DATA_FOLDER + "top100/"
PATH_TO_TOP_100_SIMILARITIES = PATH_TO_DATA_FOLDER + "top100_similarities/"

# similarity csv
PATH_TO_RAW_SIMILARITY = PATH_TO_DATA_FOLDER + "all_similarities.csv"
PATH_TO_SIMILARITY_MEAN: str = PATH_TO_DATA_FOLDER + "clean_similarity.csv"
PATH_TO_SIMILARITY_MPG: str = PATH_TO_DATA_FOLDER + "similarity_mpg.csv"
PATH_TO_SIMILARITY_MP2G: str = PATH_TO_DATA_FOLDER + "similarity_mp2g.csv"
PATH_TO_SIM_100_MPG: str = PATH_TO_TOP_100 + "similarities_mpg.csv"  # similarities mpg for top 100 movies
PATH_TO_SIM_100_MPG_SIMILARITIES: str = PATH_TO_TOP_100_SIMILARITIES + "similarities_mpg.csv"
PATH_TO_SIM_100_MP2G_SIMILARITIES: str = PATH_TO_TOP_100_SIMILARITIES + "similarities_mp2g.csv"
PATH_TO_SIM_100_MP2G: str = PATH_TO_TOP_100 + "similarities_mp2g.csv"  # similarities mp2g for top 100 movies

# movie ids csv
PATH_TO_ALL_MOVIES_ID: str = PATH_TO_DATA_FOLDER + "all_movies_ids.csv"
PATH_TO_TOP_10_MOVIES_ID: str = PATH_TO_DATA_FOLDER + "top_10_movies_ids.csv"
PATH_TO_TOP_100_MOVIES_ID: str = PATH_TO_TOP_100 + "top_100_movies_ids.csv"
PATH_TO_TOP_100_SIMILARITIES_MOVIES_ID: str = PATH_TO_TOP_100_SIMILARITIES + "movies_ids.csv"

# lists of movies
PATH_TO_MOVIES_LIST_FOLDER: str = PATH_TO_DATA_FOLDER + "lists_of_movies/"

# movie ids conversion
PATH_TO_LINK: str = PATH_TO_DATA_FOLDER + "links.csv"

# path to movie JSON
PATH_TO_JSON = PATH_TO_DATA_FOLDER + "extracted_content_ml-latest/"
PATH_TO_TOP_100_JSON = PATH_TO_TOP_100 + "movies/"
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
        # if rows_read % 100000 == 0:
        #     print(f"{rows_read} rows read")
        rows_read += 1
        if similarity_row.movie1 in list_of_movies and similarity_row.movie2 in list_of_movies:
            # similarity of 2 movies in list_of_movies
            movies_similarities = movies_similarities.append(similarity_row)

    movies_similarities.movie1 = movies_similarities.movie1.astype(int)  # movie1 treated as int
    movies_similarities.movie2 = movies_similarities.movie2.astype(int)  # movie2 treated as int
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


def read_movies_from_csv(path_to_movie_ids: str, path_to_json: str) -> List[DataFrame]:
    """
    Returns dataframe movies whose ids are in path
    :param path_to_movie_ids: path where movie ids are
    :@param path_to_json: path where jsons are
    :return: Dataframe of movies
    """
    movie_ids: List[int] = read_movie_ids_from_csv(path_to_movie_ids)
    return get_movies_by_id(movie_ids, path_to_json)


def get_similar_movies(movies_dataframes: List[DataFrame]) -> List[int]:
    """
    Returns list of movie ids who are inserted in the column "recommendations" for the movies
    passed as movies_dataframes
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
    movie_ids_movieId: List[int] = list(rows_of_movies['movieId'])
    return movie_ids_movieId


def get_movies_by_id(list_of_movies: List[int], path_to_movies: str) -> List[DataFrame]:
    """
    Return dataframe of movies whose ids were passed by list_of_movies
    @param list_of_movies: List of movies to get
    @param path_to_movies: path where json are
    """
    movies: List[DataFrame] = []
    for movie_id in list_of_movies:
        movies.append(get_movie_from_json_folder(movie_id, path_to_movies))
    return movies


def get_movie_dataframe_from_id(movie_id: int) -> DataFrame:
    """
    Return dataframe of movie
    :param movie_id: id of movie
    :return: dataframe of movie
    """
    path: str = PATH_TO_JSON + str(movie_id) + ".json"
    return pd.read_json(path)


def get_movie_from_json_folder(movie_id: int, path: str) -> DataFrame:
    """
    Return dataframe of movie reading the path
    @param movie_id: id of movie
    @param path: path where the JSON folder is
    """
    path: str = path + str(movie_id) + ".json"
    df_movie: Optional[DataFrame] = None
    try:
        df_movie = pd.read_json(path)
    except ValueError:
        print(f"File not found: \n\t{path}")
    return df_movie


def get_movie_name(movie: DataFrame) -> str:
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
    movie_df: DataFrame = get_movie_dataframe_from_id(movie_id)
    print("")
    print("Movie: " + get_movie_name(movie_df))
    for column in columns_to_print:
        print("\t" + column)
        print(f"\t{movie_df['tmdb'][column]}")
    print("")


def get_ILS(similarity_measures: pd.DataFrame, list_of_movies: List[int], method: str) -> Optional[float]:
    """
    Returns ILS value for the list_of_movies using the similarity_measures
    :param similarity_measures: dataframe of similarity measurements
    :param list_of_movies: list of movies ids
    :param method: method to compute ILS
    :return: ILS value for list_of_movies using the similarities in similarity_measures, returns none if
    similarity_measures is empty
    """
    # get similarity Dataframe for the movies in list_of_movies
    similarities_of_movies: DataFrame = get_similarities_of_movies(similarity_measures, list_of_movies)

    if similarities_of_movies.shape[0] <= 0:  # the similarity dataframe is empty
        print("There are no similarities for the movies, ILS not computable")
        return None

    ILS: float = 0
    if method == "mean":
        ILS = similarities_of_movies['similarity'].sum()
    elif method == "plot":
        for movie in list_of_movies:
            print_movie_with_info(movie, ["overview"])
        ILS = similarities_of_movies['Plot:LDA'].sum()
    elif method == "plot2":
        for movie in list_of_movies:
            print_movie_with_info(movie, ["overview"])
        ILS = similarities_of_movies['Plot:cos'].sum()
    elif method == "genre":
        for movie in list_of_movies:
            print_movie_with_info(movie, ["genres"])
        ILS = similarities_of_movies['Genre:Jacc'].sum()
    elif method == "plot-genre":
        for movie in list_of_movies:
            print_movie_with_info(movie, ["genres", "overview"])
        # mean of plot and genre
        ILS = (similarities_of_movies['Plot:LDA'].sum() + similarities_of_movies['Genre:Jacc'].sum()) / 2
    elif method == "plot2-genre":
        for movie in list_of_movies:
            print_movie_with_info(movie, ["genres", "overview"])
        # mean of plot and genre
        ILS = (similarities_of_movies['Plot:cos'].sum() + similarities_of_movies['Genre:Jacc'].sum()) / 2

    # we normalize using the number of similarities
    ILS_normalized: Optional[float]
    if similarities_of_movies.shape[0] > 0:  # there are similarities for the movies
        ILS_normalized = ILS / similarities_of_movies.shape[0]
    else:  # there are no similarities, ILS cannot be computed
        ILS_normalized = None
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


def print_names_of_movies(movie_ids: List[int], path_to_movies) -> None:
    """
    Prints names of movies whose ids are in movie_ids
    @param path_to_movies: path where movies json are
    @param movie_ids: ids of movies
    """
    movies_dataframes: List[DataFrame] = get_movies_by_id(movie_ids, path_to_movies)
    for index, movie in enumerate(movies_dataframes, start=0):
        movie_name: str = get_movie_name(movie)
        print("\t" + movie_name + ", id = " + str(movie_ids[index]))


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

    print_ILS_measures(sample_list_of_movies, similarities_df, PATH_TO_TOP_100_JSON)


def get_and_print_ILS_measurements(movies: List[int], similarity_df: DataFrame, path_to_movies: str) -> \
        Optional[Dict[str, float]]:
    """
    Returns a dict with keys ['m', 'p', 'g', 'pg'], which are the ILS measures for the movies by
    ['mean similarity', 'plot', 'genre', 'plot-genre']. Returns None if there are no measurements
    @param movies: list of movies to compute similarity for
    @type movies: List[int]
    @param similarity_df: dataframe of similarities
    @type similarity_df: DataFrame
    @param path_to_movies: path to movies jsons
    @type path_to_movies: str
    """
    dict_of_similarities: Dict[str, float] = {}
    print("movies: ")
    print_names_of_movies(movies, path_to_movies)

    ILS_m: Optional[float] = get_ILS(similarity_df, movies, "mean")
    if ILS_m is None:
        return None

    print("ILS using mean of similarities: ")
    print(ILS_m)
    dict_of_similarities['m'] = ILS_m

    ILS_p: float = get_ILS(similarity_df, movies, "plot")
    print("ILS using mean of Plot:LDA: ")
    print(ILS_p)
    dict_of_similarities['p'] = ILS_p

    ILS_p2: float = get_ILS(similarity_df, movies, "plot2")
    print("ILS using mean of Plot:cos: ")
    print(ILS_p2)
    dict_of_similarities['p2'] = ILS_p2

    ILS_g: float = get_ILS(similarity_df, movies, "genre")
    print("ILS using mean of Genre:JACC: ")
    print(ILS_g)
    dict_of_similarities['g'] = ILS_g

    ILS_pg: float = get_ILS(similarity_df, movies, "plot-genre")
    print("ILS using mean of Plot:LDA and Genre: ")
    print(ILS_pg)
    dict_of_similarities['pg'] = ILS_pg

    ILS_p2g: float = get_ILS(similarity_df, movies, "plot2-genre")
    print("ILS using mean of Plot:cos and Genre: ")
    print(ILS_p2g)
    dict_of_similarities['p2g'] = ILS_p2g

    return dict_of_similarities


def print_ILS_measures(movies: List[int], similarity_df: DataFrame, path_to_movies: str) -> None:
    print("movies: ")
    print_names_of_movies(movies, path_to_movies)

    ILS_m: Optional[float] = get_ILS(similarity_df, movies, "mean")
    if ILS_m is None:
        return  # there are no similarities for the movies

    print("ILS using mean of similarities: ")
    print(ILS_m)

    ILS_p: float = get_ILS(similarity_df, movies, "plot")
    print("ILS using mean of Plot:LDA: ")
    print(ILS_p)

    ILS_g: float = get_ILS(similarity_df, movies, "genre")
    print("ILS using mean of Genre:JACC: ")
    print(ILS_g)

    ILS_pg: float = get_ILS(similarity_df, movies, "plot-genre")
    print("ILS using mean of Plot and Genre: ")
    print(ILS_pg)


def get_integer(value: any) -> Optional[int]:
    """
    Returns int conversion of value if it is int, None otherwise
    @param value: Value to check
    @type value: any
    """
    try:
        value_to_int: int = int(value)
        return value_to_int
    except ValueError:
        return None


def print_ILS_from_ids() -> None:
    """
    Takes list of ids as input and prints ILS for the corresponding list
    """
    available_movie_ids: List[int] = read_movie_ids_from_csv(PATH_TO_TOP_100_SIMILARITIES_MOVIES_ID)

    while True:
        done: bool = False
        set_of_movies: Set[int] = set()
        while not done:
            print("enter movie ids to get the ILS, enter -1 to stop")
            value_inserted = input()  # get input of user
            value_to_int: Optional[int] = get_integer(value_inserted)
            if value_to_int is None:
                print("Please enter an integer")
            elif value_to_int == -1:
                if len(set_of_movies) <= 0:  # user pressed -1 without selecting any movie
                    return
                done = True
            elif value_to_int in available_movie_ids:
                set_of_movies.add(value_to_int)
                print("Movie entered correctly, enter -1 to get ILS")
            else:  # value inserted is int but does not correspond to a movieId
                print(f"The id {value_to_int} is not a valid id. Try again")
        print("Finished to enter ids, list:")
        print(set_of_movies)
        list_of_movies: List[int] = list(set_of_movies)

        print("Computing_ILS")
        similarity_df: DataFrame = get_similarity_dataframe(PATH_TO_SIM_100_MPG_SIMILARITIES)
        print_ILS_measures(list_of_movies, similarity_df, PATH_TO_TOP_100_SIMILARITIES_JSON)


def plot_ILS_lists(df_ILS_lists: List[DataFrame], ILS_measures: List[str]) -> None:
    """
    Plots the ILS by all ILS_measures for the dataframes in df_ILS_lists
    @param ILS_measures: name of measures to consider, these names should be columns in the dfs in df_ILS_lists.
    @type ILS_measures: List[str]
    @param df_ILS_lists: Dataframes of columns ['ids', 'm', 'p', 'g', 'pg'] which represent the mean
    similarity (m), plot similarity (p), genre similarity (g), and mean of genre and plot (pg) ILS for the movies with
    ids 'ids'.
    @type df_ILS_lists: DataFrame
    """
    list_of_list_items_number: List[int] = []  # list of indexes for lists
    for df in df_ILS_lists:
        list_of_list_items_number.append(df.shape[0])  # set number of movies for lists

    for measure in ILS_measures:
        print(f"Plot of ILS measure = {measure}")
        for index, df in enumerate(df_ILS_lists):
            # scatter plot ILS measure for every list of movies in the dataframe
            plt.scatter(x=range(0, list_of_list_items_number[index]), y=df[measure])
            print(f"Lists type {index}:")
            print(f"\tMean = {df[measure].mean()}")  # get mean of measure for lists
            print(f"\tStd = {df[measure].std()}")  # get standard deviation for lists
        plt.show()

    # print("Plot of ILS mean similarity")
    # for index, df in enumerate(df_ILS_lists):
    #     plt.scatter(x=range(0, list_of_list_items_number[index]), y=df['m'])  # scatter plot every list of movies
    #     print(f"List {index}:")
    #     print(f"\tMean = {df['m'].mean()}")
    #     print(f"\tStd = {df['m'].std()}")
    # plt.show()
    #
    # print("Plot of ILS by plot")
    # for index, df in enumerate(df_ILS_lists):
    #     plt.scatter(x=range(0, list_of_list_items_number[index]), y=df['p'])  # scatter plot every list of movies
    # plt.show()
    #
    # print("Plot of ILS by genre")
    # for index, df in enumerate(df_ILS_lists):
    #     plt.scatter(x=range(0, list_of_list_items_number[index]), y=df['g'])  # scatter plot every list of movies
    # plt.show()
    #
    # print("Plot of ILS by plot and genre")
    # for index, df in enumerate(df_ILS_lists):
    #     plt.scatter(x=range(0, list_of_list_items_number[index]), y=df['pg'])  # scatter plot every list of movies
    # plt.show()


def print_similar_movies_ILS() -> None:
    id_movies_top_100: List[int] = read_movie_ids_from_csv(PATH_TO_TOP_100_MOVIES_ID)
    similarity_df: DataFrame = get_similarity_dataframe(PATH_TO_SIM_100_MPG_SIMILARITIES)
    df_ILS_lists: DataFrame = DataFrame()  # dataframe of ils measurements for every list of movies

    index_of_lists: int = 0

    while True:
        print("Press enter to sample a movie and look for ILS of similarities. Enter -1 to exit")
        inserted_input = input()
        if get_integer(inserted_input) == -1:
            break

        movie_id: int = random.sample(id_movies_top_100, 1)[0]
        movie_df: DataFrame = get_movie_dataframe_from_id(movie_id)
        movie_name: str = get_movie_name(movie_df)
        print(f"sampled movie_id: {movie_id}, name: {movie_name}")

        movies_similar_tmdb: List[int] = get_similar_movies([movie_df])
        movies_plus_similar_movieId: List[int] = convert_tbdb_to_movieId(movies_similar_tmdb)
        movies_plus_similar_movieId.append(movie_id)
        for movie_id in movies_plus_similar_movieId:
            print(f"{movie_id},", end="")

        ils_measurements: Optional[Dict[str, any]] = \
            get_and_print_ILS_measurements(movies_plus_similar_movieId,
                                           similarity_df, PATH_TO_TOP_100_SIMILARITIES_JSON)
        if ils_measurements is None:  # there are no similarities for the list
            print("It it not possible to compute similarity for the selected movies. Try again")
        else:  # ils was computed successfully
            print(f"ILS values of list {index_of_lists} finished")
            print_names_of_movies(movies_plus_similar_movieId, PATH_TO_TOP_100_SIMILARITIES_JSON)
            index_of_lists += 1
            print("------------")
            ils_measurements['ids'] = movies_plus_similar_movieId  # add ids of movies to dict
            df_ILS_lists = df_ILS_lists.append(ils_measurements, ignore_index=True)

    if index_of_lists > 0:  # there is at least an ILS to plot
        plot_ILS_lists([df_ILS_lists], ['m', 'p', 'g', 'pg'])

    print("random_movies_ILS done")


def print_random_movies_ILS() -> None:
    """
    Asks the user a number n, then computes ILS taking n random movies from top 100
    """
    print("random_movies_ILS starts...")
    id_movies_top_100: List[int] = read_movie_ids_from_csv(PATH_TO_TOP_100_MOVIES_ID)
    similarity_df: DataFrame = get_similarity_dataframe(PATH_TO_SIM_100_MPG_SIMILARITIES)
    df_ILS_lists: DataFrame = DataFrame()  # dataframe of ils measurements for every list of movies

    index_of_lists: int = 0

    while True:
        print("Enter the number of random movies to insert in the list or enter -1 to stop")
        number_of_movies: Optional[int] = get_integer(input())
        if number_of_movies is None:  # value inserted is not an int
            print("Please enter an integer")
        elif number_of_movies > 0:
            random_ids: List[int] = random.sample(id_movies_top_100, number_of_movies)
            for movie_id in random_ids:
                print(f"{movie_id},", end="")
            # dict of ils measurements, keys = ['m', 'p', 'g', 'pg']
            ils_measurements: Optional[Dict[str, any]] = \
                get_and_print_ILS_measurements(random_ids, similarity_df, PATH_TO_TOP_100_JSON)
            if ils_measurements is None:  # there are no similarities for the list
                print("It it not possible to compute similarity for the selected movies. Try again")
            else:  # ils was computed successfully
                print(f"ILS values of list {index_of_lists} finished")
                print_names_of_movies(random_ids, PATH_TO_TOP_100_JSON)
                index_of_lists += 1
                print("------------")
                ils_measurements['ids'] = random_ids  # add ids of movies to dict
                df_ILS_lists = df_ILS_lists.append(ils_measurements, ignore_index=True)
        else:  # value inserted is negative
            break
    if index_of_lists > 0:
        plot_ILS_lists([df_ILS_lists], ['m', 'p', 'g', 'pg'])

    print("random_movies_ILS done")


def read_lists_of_int_from_csv(path: str) -> List[List[int]]:
    """
    Reads the csv and returns the corresponding lists of ints. Every line of the csv represent a list of ints, separated
    by comma.
    @param path: path to the csv containing the lists
    @type path: str
    """
    lists_of_ints: List[List[int]] = []
    with open(path) as f:
        for line_list in f:
            stripped_line_list: str = line_list.strip()  # remove new line
            line_to_list_str: List[str] = stripped_line_list.split(",")
            list_to_list_int: List[int] = [int(i) for i in line_to_list_str]
            lists_of_ints.append(list_to_list_int)
    return lists_of_ints


def get_dataframe_of_movie_lists(lists_of_movies: List[List[int]],
                                 similarity_df: DataFrame,
                                 path_to_similarities: str) -> DataFrame:
    """
    Returns a dataframe, where every row contains the ids of the list and the ILS measurements [mean, plot, genre,
    plot-genre]
    @param path_to_similarities: path to json of movies
    @type path_to_similarities: str
    @param similarity_df: DataFrame of similarities
    @type similarity_df: DataFrame
    @param lists_of_movies: lists of movies to compute similarity for
    @type lists_of_movies: List[List[int]]
    """
    df_ILS_lists: DataFrame = DataFrame()  # dataframe of ils measurements for every list of movies

    index_of_lists: int = 0

    for list_of_movies in lists_of_movies:
        ils_measurements: Optional[Dict[str, any]] = \
            get_and_print_ILS_measurements(list_of_movies,
                                           similarity_df, path_to_similarities)
        if ils_measurements is None:  # there are no similarities for the list
            print("It it not possible to compute similarity for the selected movies. Try again")
        else:  # ils was computed successfully
            print(f"ILS values of list {index_of_lists} finished")
            print_names_of_movies(list_of_movies, path_to_similarities)
            index_of_lists += 1
            print("------------")
            ils_measurements['ids'] = list_of_movies  # add ids of movies to dict
            df_ILS_lists = df_ILS_lists.append(ils_measurements, ignore_index=True)

    return df_ILS_lists


def print_lists_in_file_ILS() -> None:
    """
    Reads the lists of movies written in data/similar_movies.csv and computes then plots ils.
    """
    similarities_top100_similarities: DataFrame = get_similarity_dataframe(PATH_TO_SIM_100_MP2G_SIMILARITIES)
    similarities_top100: DataFrame = get_similarity_dataframe(PATH_TO_SIM_100_MP2G)
    # read lists of movies from file
    lists_of_similar_movies: List[List[int]] = \
        read_lists_of_int_from_csv(PATH_TO_MOVIES_LIST_FOLDER + "similar_movies.csv")
    lists_of_random_movies: List[List[int]] = \
        read_lists_of_int_from_csv(PATH_TO_MOVIES_LIST_FOLDER + "random_movies.csv")
    lists_of_hand_made_movies: List[List[int]] = \
        read_lists_of_int_from_csv(PATH_TO_MOVIES_LIST_FOLDER + "hand_made_movies.csv")
    # dataframe of ILS measurements for lists of similar movies
    df_ILS_similar_movies: DataFrame = get_dataframe_of_movie_lists(lists_of_similar_movies,
                                                                    similarities_top100_similarities,
                                                                    PATH_TO_TOP_100_SIMILARITIES_JSON)

    # dataframe of ILS measurements for lists of random movies
    df_ILS_random_movies: DataFrame = get_dataframe_of_movie_lists(lists_of_random_movies,
                                                                   similarities_top100,
                                                                   PATH_TO_TOP_100_JSON)

    df_ILS_hand_made_movies: DataFrame = get_dataframe_of_movie_lists(lists_of_hand_made_movies,
                                                                      similarities_top100,
                                                                      PATH_TO_TOP_100_JSON)

    plot_ILS_lists([df_ILS_similar_movies, df_ILS_random_movies, df_ILS_hand_made_movies],
                   ['m', 'p', 'p2', 'g', 'pg', 'p2g'])

    print("lists_in_file_ILS done")
