from src.lists import MoviesLists, ListsNames

if __name__ == "__main__":
    hand_made_movies: MoviesLists = MoviesLists(ListsNames.HAND_MADE)
    hand_made_movies.plot()
