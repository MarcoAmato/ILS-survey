from src.lists import maximize_similarity_neighbors_lists, MoviesLists


def test_maximize_similarity_neighbors_lists():
    return maximize_similarity_neighbors_lists(MoviesLists.HAND_MADE)


def test_set_lists():
    list = test_maximize_similarity_neighbors_lists()
    MoviesLists.MAX_NEIGHBOURS.set_lists(list)


if __name__ == "__main__":
    test_set_lists()
