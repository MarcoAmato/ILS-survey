# lists of movies folders
import itertools
import os
from enum import Enum
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from src.pre_computation import write_movie_ids_to_csv, write_similarities_of_movies
from src.similarities_util import PATH_TO_DATA_FOLDER, read_lists_of_int_from_csv, matrix_to_list, \
    get_dataframe_of_movie_lists, PATH_TO_JSON, PATH_TO_SIMILARITY_MP2G, SimilarityMethod, plot_ILS_with_label, \
    read_movie_ids_from_csv, get_random_movies, get_ILS

PATH_TO_MOVIES_LIST_FOLDER: str = PATH_TO_DATA_FOLDER + "lists_of_movies/"


class ListsNames(Enum):  # enum of possible lists
    HAND_MADE = "hand_made/"
    HAND_MADE_CLUSTERS = "hand_made_clusters/"
    INCREASING_ILD = "increasing_ILD/"
    BATMAN = "batman/"
    RECOMMENDATIONS = "recommendations/"
    MAX_NEIGHBOURS = "max_neighbours/"
    MIN_NEIGHBOURS = "min_neighbours/"
    RANDOM_10 = "random_10/"
    RANDOM_3 = "random_3/"
    TEST = "test/"


class MoviesLists:
    list_name: str
    path_to_folder: str
    lists: List[List[int]]
    ids: List[int]
    similarities: DataFrame
    dataframe_lists: DataFrame
    labels: List[str]

    def __init__(self, list_name: ListsNames,
                 lists: Optional[List[List[int]]] = None,
                 labels: Optional[List[str]] = None):
        self.list_name = list_name.name
        self.path_to_folder = PATH_TO_MOVIES_LIST_FOLDER + list_name.value

        if lists is not None:
            self.write_lists(lists)  # write list on dataframe
            self.lists = lists  # set list
            self.__set_label(labels)  # set labels
            self.pre_compute()  # recompute data
        else:
            self.lists = read_lists_of_int_from_csv(self.get_path_lists())  # lists should be taken from csv
            self.__set_label(labels)

        try:
            self.__set_pre_computed_data()  # if the precomputed data is not available error is thrown
        except FileNotFoundError:
            print("Computing lists data...")  # so we compute it before going forward
            self.pre_compute()
            self.__set_pre_computed_data()

    def get_path_lists(self) -> str:
        return self.path_to_folder + "lists.csv"

    def get_path_similarities(self) -> str:
        return self.path_to_folder + "similarities.csv"

    def get_path_ids(self) -> str:
        return self.path_to_folder + "ids.csv"

    def get_path_dataframe_lists(self) -> str:
        return self.path_to_folder + "dataframe_lists.csv"

    def __set_pre_computed_data(self):
        self.ids = read_movie_ids_from_csv(self.get_path_ids())
        self.similarities = pd.read_csv(self.get_path_similarities())
        self.dataframe_lists = pd.read_csv(self.get_path_dataframe_lists())

    def __set_label(self, labels=None):
        if labels is not None:
            self.labels = labels
        else:
            self.labels = list(range(0, len(self.lists)))  # if labels not set insert number placeholder

    def write_lists(self, list_of_lists: List[List[int]]) -> None:
        if not os.path.exists(self.get_path_lists()):  # if lists.csv does not exist
            open(self.get_path_lists(), "x")  # create file

        with open(self.get_path_lists(), "w") as f:  # open file
            for list in list_of_lists:  # for each list
                for item in list[:-1]:  # do for every item except last
                    f.write(f"{item}, ")
                f.write(f"{list[-1]}")  # don't write comma for last item
                f.write("\n")  # write list and new line

        self.lists = list_of_lists

    def write_ids(self):
        list_of_ids: List[int] = list(set(matrix_to_list(self.lists)))  # convert to list of ids
        write_movie_ids_to_csv(list_of_ids, self.get_path_ids())

        self.ids = list_of_ids

    def write_similarities(self):
        similarities = write_similarities_of_movies(PATH_TO_SIMILARITY_MP2G, movies=self.ids,
                                                    path_to_write=self.get_path_similarities())

        self.similarities = similarities

    def write_dataframe_lists(self):
        dataframe_lists = get_dataframe_of_movie_lists(self.lists, self.similarities, PATH_TO_JSON)
        dataframe_lists['label'] = self.labels  # set label
        dataframe_lists.to_csv(self.get_path_dataframe_lists())

        self.dataframe_lists = dataframe_lists

    def pre_compute(self) -> None:
        if len(self.lists) > 0:  # if lists exist
            self.write_ids()
            self.write_similarities()
            self.write_dataframe_lists()
        else:
            raise ValueError("lists.csv is empty, pre computation failed")

    def plot(self) -> None:
        """
        Plots the list. This list should have been pre computed by calling pre_compute().
        """
        print("print_lists_in_file_ILS starts...")

        plot_ILS_with_label(self.dataframe_lists, ['m'])

        print("print_lists_in_file_ILS done")


class Random10Lists(MoviesLists):
    def __init__(self, list_name: ListsNames, recreate: bool = False):
        if list_name.name == ListsNames.RANDOM_10.name:
            if recreate:
                new_random_lists = []
                for _ in itertools.repeat(None, 10):  # for 10 times (slightly faster than 'for i in range')
                    new_random_lists.append(get_random_movies(7))
                super().__init__(list_name, lists=new_random_lists)
            else:
                super().__init__(list_name)
        else:
            raise ValueError("RandomMoviesLists list_name should be RANDOM_10")

    def write_top_middle_bottom_lists(self) -> MoviesLists:
        """
        Writes the top, middle, bottom list from the lists ordered by ILS, to the RANDOM_3 list and returns object
        """
        df_sorted = self.dataframe_lists.sort_values("m")  # get lists sorted by mean ILS
        list_of_ids = df_sorted.ids.tolist()  # list of ids sorted by ascending mean (elements are strings)
        list_of_ids_int: List[List[int]] = []  # need to convert elements to lists of ints

        for list_str in list_of_ids:
            list_converted = convert_string_to_list_of_int(list_str)
            list_of_ids_int.append(list_converted)

        bottom_list = list_of_ids_int[0]
        middle_list = list_of_ids_int[(len(list_of_ids_int) - 1) // 2]  # take middle item in list
        top_list = list_of_ids_int[-1]

        top_middle_bottom_list: List[List[int]] = [bottom_list, middle_list, top_list]  # create new list of lists

        random_3 = MoviesLists(ListsNames.RANDOM_3, lists=top_middle_bottom_list, labels=["bottom", "middle", "top"])

        return random_3


def convert_string_to_list_of_int(string: str) -> List[int]:
    brackets_removed = string.strip("[]")  # remove brackets
    string_to_list = brackets_removed.split(",")  # list of strings where string is the number
    list_to_int = list(map(int, string_to_list))  # convert elements to actual ints
    return list_to_int


def maximize_similarity_neighbors_lists(movies_list: MoviesLists) -> List[List[int]]:
    """
    Returns the list ListNames, where every list is ordered by maximizing the ILS of neighbours
    @param movies_list: list to order
    @type movies_list: MoviesLists
    @return: the list ListNames, where every list is ordered by maximizing the ILS of neighbours
    @rtype: List[List[int]]
    """
    list_of_lists: List[List[int]] = movies_list.lists
    similarities: DataFrame = movies_list.similarities  # get similarities for list_of_lists

    max_sim_lists: List[List[int]] = []

    for list in list_of_lists:
        max_sim_list: List[int] = []
        remaining_items: List[int] = list.copy()
        max_sim_list.append(list[0])  # add first movie to max_sim_list

        # print(max_sim_list)
        # print(remaining_items)

        del remaining_items[0]  # remove first item of list from remaining items
        for items_added in range(1, len(list)):  # iterate list from second item to last
            # print(f"adding item {items_added}")
            max_sim_list = add_item_to_list_max_ILS(max_sim_list, remaining_items, similarities, SimilarityMethod.MEAN)
            remaining_items.remove(max_sim_list[-1])  # remove last item added to max_sim_list from remaining_items
        max_sim_lists.append(max_sim_list)  # add max_sim_list to max_sim_lists

    return max_sim_lists


def minimize_similarity_neighbors_lists(movies_list: MoviesLists) -> List[List[int]]:
    """
    Returns the list ListNames, where every list is ordered by minimizing the ILS of neighbours
    @param movies_list: list to order
    @type movies_list: MoviesLists
    @return: the list ListNames, where every list is ordered by minimizing the ILS of neighbours
    @rtype: List[List[int]]
    """
    list_of_lists: List[List[int]] = movies_list.lists
    similarities: DataFrame = movies_list.similarities  # get similarities for list_of_lists

    max_sim_lists: List[List[int]] = []

    for list in list_of_lists:
        min_sim_list: List[int] = []
        remaining_items: List[int] = list.copy()
        min_sim_list.append(list[0])  # add first movie to max_sim_list

        # print(max_sim_list)
        # print(remaining_items)

        del remaining_items[0]  # remove first item of list from remaining items
        for items_added in range(1, len(list)):  # iterate list from second item to last
            # print(f"adding item {items_added}")
            min_sim_list = add_item_to_list_min_ILS(min_sim_list, remaining_items, similarities, SimilarityMethod.MEAN)
            remaining_items.remove(min_sim_list[-1])  # remove last item added to max_sim_list from remaining_items
        max_sim_lists.append(min_sim_list)  # add max_sim_list to max_sim_lists

    return max_sim_lists


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


def add_item_to_list_min_ILS(list_to_minimize: List[int],
                             items_to_choose: List[int],
                             similarity_df: DataFrame,
                             similarity_method: SimilarityMethod
                             ) -> Optional[List[int]]:
    """
    Returns a list got by adding one element from items_to_choose to list_to_minimize. The item is chosen in order to
    have ILS minimized.
    @param similarity_method: Method to compute similarity
    @type similarity_method: SimilarityMethod
    @param similarity_df: dataframe of similarity to compute ILS
    @type similarity_df: DataFrame
    @param items_to_choose: list of items to choose in order to minimize ILS of list
    @type items_to_choose: List[int]
    @param list_to_minimize: list to be minimized
    @type list_to_minimize: List[int]
    """
    min_ILS: float = 2  # min ILS value by adding the remaining items
    min_item: Optional[int] = None
    for item in items_to_choose:
        list_minimized_plus_item = list_to_minimize.copy()
        list_minimized_plus_item.append(item)
        ils_i: float = get_ILS(similarity_df, list_minimized_plus_item, similarity_method.value)
        if ils_i is not None and ils_i < min_ILS:  # if ILS is not computable, skip item
            min_ILS = ils_i
            min_item = item
    if min_item is not None:
        list_minimized = list_to_minimize + [min_item]  # add item that keeps ils the highest
        return list_minimized
    else:
        # if there are no similarity for the movies, the first item is added
        list_to_minimize.append(items_to_choose[0])
        return list_to_minimize
