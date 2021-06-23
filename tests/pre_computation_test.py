from src.pre_computation import write_movies_info_file, COLUMNS_MOVIES_PLOT_GENRE, maximize_similarity_neighbors_lists
from src.similarities_util import PATH_TO_TOP_100_JSON, PATH_TO_DESCRIPTION_TOP_100, ListNames


def test_maximize_similarity_neighbors_lists():
    print(maximize_similarity_neighbors_lists(ListNames.HAND_MADE))


if __name__ == "__main__":
    test_maximize_similarity_neighbors_lists()
