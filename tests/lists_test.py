import os
from typing import List

from src.lists import MoviesLists, ListsNames, PATH_TO_MOVIES_LIST_FOLDER, Random10Lists, \
    minimize_similarity_neighbors_lists, maximize_similarity_neighbors_lists, maximize_distance_of_first

PATH_TO_TEST_LIST = PATH_TO_MOVIES_LIST_FOLDER + "test/"
PATH_TO_SIMILARITIES_TEST = PATH_TO_TEST_LIST + "similarities.csv"
PATH_TO_IDS_TEST = PATH_TO_TEST_LIST + "ids.csv"
PATH_TO_DATAFRAME_LISTS_TEST = PATH_TO_TEST_LIST + "dataframe_lists.csv"


def test_plot():
    test_list = MoviesLists(ListsNames.TEST)
    test_list.plot()


def test_pre_compute_lists():
    # remove pre computed csv
    if os.path.exists(PATH_TO_SIMILARITIES_TEST):
        os.remove(PATH_TO_SIMILARITIES_TEST)
    if os.path.exists(PATH_TO_IDS_TEST):
        os.remove(PATH_TO_IDS_TEST)
    if os.path.exists(PATH_TO_DATAFRAME_LISTS_TEST):
        os.remove(PATH_TO_DATAFRAME_LISTS_TEST)
    test_plot()


def test_write_top_middle_bottom():
    random_10 = Random10Lists(ListsNames.RANDOM_10, recreate=True)
    random_10.write_top_middle_bottom_lists()


def test_minimize_similarity_recommendations():
    recommendations: MoviesLists = MoviesLists(ListsNames.RECOMMENDATIONS)
    list: List[List[int]] = minimize_similarity_neighbors_lists(recommendations)
    min_neighbours: MoviesLists = MoviesLists(ListsNames.MIN_NEIGHBOURS, list)
    min_neighbours.plot()


def test_maximize_similarity_recommendations():
    recommendations: MoviesLists = MoviesLists(ListsNames.RECOMMENDATIONS)
    list: List[List[int]] = maximize_similarity_neighbors_lists(recommendations)
    max_neighbours: MoviesLists = MoviesLists(ListsNames.MAX_NEIGHBOURS, list)
    max_neighbours.plot()


def test_maximize_distance_to_first_recommendations():
    maximized: List[List[int]] = maximize_distance_of_first(MoviesLists(ListsNames.RECOMMENDATIONS))
    MoviesLists(ListsNames.MAX_DIST_FIRST).plot()


if __name__ == "__main__":
    test_maximize_distance_to_first_recommendations()
