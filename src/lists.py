# lists of movies folders
import os
from enum import Enum
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from src.pre_computation import write_movie_ids_to_csv, write_similarities_of_movies, add_item_to_list_max_ILS
from src.similarities_util import PATH_TO_DATA_FOLDER, read_lists_of_int_from_csv, matrix_to_list, \
    get_dataframe_of_movie_lists, PATH_TO_JSON, PATH_TO_SIMILARITY_MP2G, SimilarityMethod, plot_ILS_with_label

PATH_TO_MOVIES_LIST_FOLDER: str = PATH_TO_DATA_FOLDER + "lists_of_movies/"


class ListsNames(Enum):  # enum of possible lists
    HAND_MADE = "hand_made/"
    HAND_MADE_CLUSTERS = "hand_made_clusters/"
    INCREASING_ILD = "increasing_ILD/"
    BATMAN = "batman/"
    MAX_NEIGHBOURS = "max_neighbours/"


class MoviesLists:
    list_name: str
    path_to_folder: str

    def __init__(self, list_name: ListsNames):
        self.list_name = list_name.name
        self.path_to_folder = PATH_TO_MOVIES_LIST_FOLDER + list_name.value

    def get_path_lists(self) -> str:
        return self.path_to_folder + "lists.csv"

    def get_path_similarities(self) -> str:
        return self.path_to_folder + "similarities.csv"

    def get_path_ids(self) -> str:
        return self.path_to_folder + "ids.csv"

    def get_path_dataframe_lists(self) -> str:
        return self.path_to_folder + "dataframe_lists.csv"

    def get_list_of_lists(self) -> List[List[int]]:
        return read_lists_of_int_from_csv(self.get_path_lists())

    def get_similarities(self) -> DataFrame:
        return pd.read_csv(self.get_path_similarities())

    def get_dataframe_lists(self) -> DataFrame:
        return pd.read_csv(self.get_path_dataframe_lists())

    def set_lists(self, list_of_lists: List[List[int]]):
        if not os.path.exists(self.get_path_lists()):  # if lists.csv does not exist
            open(self.get_path_lists(), "x")  # create file

        with open(self.get_path_lists(), "w") as f:  # open file
            for list in list_of_lists:  # for each list
                for item in list[:-1]:  # do for every item except last
                    f.write(f"{item}, ")
                f.write(f"{list[-1]}")  # don't write comma for last item
                f.write("\n")  # write list and new line

    def set_ids_from_lists(self):
        list_of_lists: List[List[int]] = self.get_list_of_lists()
        list_of_ids: List[int] = list(set(matrix_to_list(list_of_lists)))  # convert to list of ids
        write_movie_ids_to_csv(list_of_ids, self.get_path_ids())

    def set_dataframe_lists(self, list_of_lists: List[List[int]], labels: Optional[List[str]] = None):
        df_ILS = get_dataframe_of_movie_lists(list_of_lists, self.get_similarities(), PATH_TO_JSON)
        if labels is not None:
            df_ILS["label"] = labels
        df_ILS.to_csv(self.get_path_dataframe_lists())

    def pre_compute(self, labels: Optional[List[str]] = None) -> None:
        if os.path.exists(self.get_path_lists()):  # if lists of list exist
            list: List[List[int]] = self.get_list_of_lists()
            self.set_ids_from_lists()
            write_similarities_of_movies(PATH_TO_SIMILARITY_MP2G, self.get_path_ids(),
                                         self.get_path_similarities())
            if labels is None:
                labels = range(len(list))  # if labels not set, labels = [0,1,..,len(list_of_lists)]
            self.set_dataframe_lists(list_of_lists=list,
                                     labels=labels)

    def plot(self) -> None:
        """
        Plots the list. This list should have been pre computed by calling pre_compute().
        """
        print("print_lists_in_file_ILS starts...")

        dataframe_lists: DataFrame = self.get_dataframe_lists()

        plot_ILS_with_label(dataframe_lists, ['m', 'g'])

        print("print_lists_in_file_ILS done")


def maximize_similarity_neighbors_lists(list_name: MoviesLists) -> List[List[int]]:
    """
    Returns the list ListNames, where every list is ordered by maximizing the ILS of neighbours
    @param list_name: list to order
    @type list_name: MoviesLists
    @return: the list ListNames, where every list is ordered by maximizing the ILS of neighbours
    @rtype: List[List[int]]
    """
    list_of_lists: List[List[int]] = list_name.get_list_of_lists()
    similarities: DataFrame = list_name.get_similarities()  # get similarities for list_of_lists

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
