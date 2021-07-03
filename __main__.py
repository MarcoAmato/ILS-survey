from typing import Optional

import src.similarities_util as sim
from src.lists import MoviesLists, ListsNames

if __name__ == "__main__":
    while True:
        print("ILS creator. \n\tEnter 1 to create lists by entering the ids of top100 popularity movies manually"
              "\n\tEnter 2 to create lists from the similarities (as reported in the json files) of a random movie "
              "in the top100"
              "\n\tEnter 3 to create a list of random items"
              "\n\tEnter 4 to create lists from data/lists_of_movies/hand_made_movies (more information of "
              "README.md) "
              "\n\tEnter 5 to create lists from data/lists_of_movies/increasing_ILD"
              "\n\tEnter 6 to create lists from data/lists_of_movies/hand_made_clusters"
              "\n\tEnter 7 to create lists from data/lists_of_movies/batman"
              "\n\tEnter -1 to exit")
        value_inserted = input()
        command_inserted: Optional[int] = sim.get_integer(value_inserted)

        if command_inserted is None:
            print(f"You inserted {value_inserted}, expected an integer command")
        elif command_inserted == 1:
            sim.print_ILS_from_ids()
        elif command_inserted == 2:
            sim.print_similar_movies_ILS()
        elif command_inserted == 3:
            sim.print_random_movies_ILS()
        elif command_inserted == 4:
            hand_made_movies: MoviesLists = MoviesLists(ListsNames.HAND_MADE)
            hand_made_movies.plot()
        elif command_inserted == 5:
            increasing_ILD: MoviesLists = MoviesLists(ListsNames.INCREASING_ILD)
            increasing_ILD.plot()
        elif command_inserted == 6:
            hand_made_clusters: MoviesLists = MoviesLists(ListsNames.HAND_MADE_CLUSTERS)
            hand_made_clusters.plot()
        elif command_inserted == 7:
            batman: MoviesLists = MoviesLists(ListsNames.BATMAN)
            batman.plot()
        elif command_inserted == -1:
            break
        else:
            print(f"Command not valid {command_inserted}")
