from typing import List

from pandas import DataFrame

from src.similarities_util import read_lists_of_int_from_csv, PATH_TO_DATA_FOLDER, get_movies_df_from_json_folder, \
    PATH_TO_TOP_100_JSON, get_movies_with_name


def test_read_lists_of_int_from_csv():
    read_lists_of_int_from_csv(PATH_TO_DATA_FOLDER + "similar_movies.csv")


def test_get_movies_df_from_json_folder():
    list_of_movies: List[DataFrame] = get_movies_df_from_json_folder(PATH_TO_TOP_100_JSON)
    print(list_of_movies)


def test_get_movies_with_name():
    movies_toy_story: List[DataFrame] = get_movies_with_name("Toy Story", PATH_TO_TOP_100_JSON)
    for movie in movies_toy_story:
        print(movie["tmdb"]["title"])


if __name__ == "__main__":
    test_get_movies_with_name()
