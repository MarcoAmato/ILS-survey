import random
from typing import List

from pandas import DataFrame

from src.similarities_util import read_movie_ids_from_csv, PATH_TO_TOP_100_MOVIES_ID, \
    get_movie_dataframe_from_id, get_similar_movies, get_similarity_dataframe, PATH_TO_SIM_100_MPG_SIMILARITIES, \
    print_ILS_measures, PATH_TO_TOP_100_SIMILARITIES_JSON, convert_tbdb_to_movieId, get_movie_name, \
    print_similar_movies_ILS, read_lists_of_int_from_csv, PATH_TO_DATA_FOLDER


def test_read_lists_of_int_from_csv():
    read_lists_of_int_from_csv(PATH_TO_DATA_FOLDER + "similar_movies.csv")


if __name__ == "__main__":
    test_read_lists_of_int_from_csv()
