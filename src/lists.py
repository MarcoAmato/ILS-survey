# lists of movies folders
import os
from enum import Enum
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from src.pre_computation import write_movie_ids_to_csv, write_similarities_of_movies, add_item_to_list_max_ILS
from src.similarities_util import PATH_TO_DATA_FOLDER, read_lists_of_int_from_csv, matrix_to_list, \
    get_dataframe_of_movie_lists, PATH_TO_JSON, PATH_TO_SIMILARITY_MP2G, SimilarityMethod, plot_ILS_with_label, \
    read_movie_ids_from_csv

PATH_TO_MOVIES_LIST_FOLDER: str = PATH_TO_DATA_FOLDER + "lists_of_movies/"


class ListsNames(Enum):  # enum of possible lists
    HAND_MADE = "hand_made/"
    HAND_MADE_CLUSTERS = "hand_made_clusters/"
    INCREASING_ILD = "increasing_ILD/"
    BATMAN = "batman/"
    MAX_NEIGHBOURS = "max_neighbours/"
    RANDOM_10 = "random_10/"


class MoviesLists:
    list_name: str
    path_to_folder: str
    lists: List[List[int]]
    ids: List[int]
    similarities: DataFrame
    dataframe_lists: DataFrame
    labels: List[str]

    def __init__(self, list_name: ListsNames, labels: Optional[List[str]] = None):
        self.list_name = list_name.name
        self.labels = labels
        self.path_to_folder = PATH_TO_MOVIES_LIST_FOLDER + list_name.value
        # if error is thrown here lists.csv was not set
        self.lists = read_lists_of_int_from_csv(self.get_path_lists())
        try:
            self.__set_pre_computed_data()  # if the precomputed data is not available error is thrown
        except IOError:
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
        dataframe_lists.to_csv(self.get_path_dataframe_lists())

        self.dataframe_lists = dataframe_lists

    def pre_compute(self) -> None:
        if self.lists is not None:  # if lists exist
            self.write_ids()
            self.write_similarities()
            if self.labels is None:
                self.labels = list(range(len(self.lists)))  # if labels not set, labels = [0,1,..,len(list_of_lists)]
            self.write_dataframe_lists()

    def plot(self) -> None:
        """
        Plots the list. This list should have been pre computed by calling pre_compute().
        """
        print("print_lists_in_file_ILS starts...")

        plot_ILS_with_label(self.dataframe_lists, ['m', 'g'])

        print("print_lists_in_file_ILS done")


class RandomMoviesLists(MoviesLists):
    def __init__(self, list_name: ListsNames):
        if list_name.name != ListsNames.RANDOM_10:
            raise ValueError("RandomMoviesLists list_name should be RANDOM_10")
        else:
            super().__init__(list_name)

    def write_top_middle_bottom_lists(self):
        """
        Writes the top, middle, bottom list from the lists ordered by ILS, to the RANDOM_3 list
        """


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

        print(max_sim_list)
        print(remaining_items)

        del remaining_items[0]  # remove first item of list from remaining items
        for items_added in range(1, len(list)):  # iterate list from second item to last
            print(f"adding item {items_added}")
            max_sim_list = add_item_to_list_max_ILS(max_sim_list, remaining_items, similarities, SimilarityMethod.MEAN)
            remaining_items.remove(max_sim_list[-1])  # remove last item added to max_sim_list from remaining_items
        max_sim_lists.append(max_sim_list)  # add max_sim_list to max_sim_lists

    return max_sim_lists
