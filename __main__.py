from typing import Optional

import src.similarities_util as sim

if __name__ == "__main__":
    while True:
        print("ILS creator. \n\tEnter 1 to create lists by entering the ids of top100 popularity movies manually"
              "\n\tEnter 2 to create lists from the similarities (as reported in the json files) of a random movie "
              "in the top100"
              "\n\tEnter 3 to create a list of random items"
              "\n\tEnter 4 to create lists from all the files in data/lists_of_movies (more information of README.md)"
              "\n\tEnter 5 to create lists from data/lists_of_movies/hand_made_movies (more information of "
              "README.md) "
              "\n\tEnter 6 to create lists from data/lists_of_movies/increasing_ILD"
              "\n\tEnter 7 to create lists from data/lists_of_movies/hand_made_clusters"
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
            sim.print_lists_in_file_ILS()
        elif command_inserted == 5:
            sim.print_pre_computed_list("hand_made")
        elif command_inserted == 6:
            sim.print_pre_computed_list("increasing_ILD")
        elif command_inserted == 7:
            sim.print_pre_computed_list("hand_made_clusters")
        elif command_inserted == -1:
            break
        else:
            print(f"Command not valid {command_inserted}")
