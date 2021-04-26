from random import sample
import pandas as pd

PATH_TO_DATA_FOLDER = "../Data/"
PATH_TO_RAW_SIMILARITY = PATH_TO_DATA_FOLDER + "pred2-incl-all_all.csv"
PATH_TO_NEW_SIMILARITY: str = PATH_TO_DATA_FOLDER + "clean_similarity.csv"  # ['movie1', movie2, 'similarity']
PATH_TO_JSON = PATH_TO_DATA_FOLDER + "extracted_content_ml-latest/"
COLUMNS_SIMILARITY = {'Title:LEV', 'Title:JW', 'Title:LCS', 'Title:BI',
                      'Title:LDA', 'Image:EMB', 'Image:BR', 'Image:SH', 'Image:CO',
                      'Image:COL', 'Image:EN', 'Plot:LDA', 'Plot:cos', 'Genre:LDA',
                      'Genre:Jacc', 'Stars:Jacc', 'Directors:Jacc', 'Date:MD', 'Tag:Genome',
                      'SVD'}
MOVIES_LIST_LENGTH = 3


def get_database_clean(num_rows: int = None) -> pd.DataFrame:
    """
    Returns the dataset where the columns are only: the two movies id and the similarity measurements.
    if 'num_rows' is specified, it is the number of similarities we are putting into the dataframe, otherwise we take
    them all
    :param num_rows: number of rows to be read
    :return: the clean dataframe
    """
    interested_columns = {"validation$r1", "validation$r2"}.union(COLUMNS_SIMILARITY)
    if num_rows is not None and num_rows >= 1:
        return pd.read_csv(PATH_TO_RAW_SIMILARITY, nrows=num_rows, sep="\t", usecols=interested_columns)
    else:
        return pd.read_csv(PATH_TO_RAW_SIMILARITY, sep="\t", usecols=interested_columns)


def get_light_dataframe(num_rows: int = None) -> pd.DataFrame:
    """
    Returns dataframe with columns ['movie1', movie2, 'similarity'], if num_rows is inserted returns first num_rows rows
    :param num_rows: number of rows to be read, if null read all the csv
    """

    if num_rows is not None and num_rows >= 1:
        return pd.read_csv(PATH_TO_NEW_SIMILARITY, nrows=num_rows)
    else:
        return pd.read_csv(PATH_TO_NEW_SIMILARITY)


# returns all the rows of the 'dataframe' where 'movie' is compared to another movie
def get_similarity_rows_of_movie(dataframe: pd.DataFrame, movie: str):
    # create empty dataframe with same structure as 'dataframe'
    sim_df_of_movie = pd.DataFrame(columns=dataframe.columns)
    for i, sim_row in dataframe.iterrows():
        # if 'movie' is in the row this is a similarity measure for 'movie'
        if sim_row["validation$r1"] == movie or sim_row["validation$r2"] == movie:
            sim_df_of_movie = sim_df_of_movie.append(sim_row, ignore_index=True)
    return sim_df_of_movie


# returns list of all the movies in 'df'
def get_all_movies(df: pd.DataFrame):
    movies = []
    for index, row in df.iterrows():
        if row.loc["movie1"] not in movies:
            movies.append(row.loc["movie1"])
        elif row.loc["movie2"] not in movies:
            movies.append(row.loc["movie2"])
    return movies


def get_film_paths(similarity_row: pd.Series):
    film1 = (similarity_row["validation$r1"])
    film2 = (similarity_row["validation$r2"])
    return [film1, film2]


def get_film(path: str):
    return pd.read_json(PATH_TO_JSON + path)


def get_name_of_film(film: str):
    film = get_film(film)
    return film["tmdb"]["title"]


def get_mean_similarity(similarity_row: pd.Series):
    similarity_values = similarity_row[COLUMNS_SIMILARITY]
    return sum(similarity_values) / len(similarity_values)


def get_ILS(similarity_measures: pd.DataFrame, list_of_movies: list[int]) -> float:
    ils: float = 0
    for movie1 in list_of_movies:
        for movie2 in list_of_movies:
            if movie1 != movie2:
                print(movie1)
                print(movie2)
                sim_movies = get_similarity(similarity_measures, movie1, movie2)
                print(sim_movies)
                ils += sim_movies
    return ils


def get_similarity(similarity_df, movie1, movie2):
    for index, similarity_row in similarity_df.iterrows():
        if (similarity_row.loc["validation$r1"] == movie1
            and similarity_row.loc["validation$r2"] == movie2) \
                or (similarity_row.loc["validation$r2"] == movie1
                    and similarity_row.loc["validation$r1"] == movie2):
            return get_mean_similarity(similarity_row)
    print("error")
    return 0


if __name__ == "__main__":
    light_df: pd.DataFrame = get_light_dataframe()
    print(light_df)
    exit()

    all_movies = get_all_movies(get_light_dataframe)

    test_list_of_movies = sample(all_movies, MOVIES_LIST_LENGTH)  # get random list of MOVIES_LIST_LENGTH movies

    test_get_ILS: float = get_ILS(similarity_dataframe, test_list_of_movies)

    print(test_get_ILS)
