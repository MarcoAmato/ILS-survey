from typing import Optional

import src.similarities_util as sim
from src.lists import MoviesLists, ListsNames

list_names = [e.name for e in ListsNames]
lists_dict = {}
for index, name in enumerate(list_names):
    lists_dict[index] = name


def print_lists():
    print("\tNumber\tName")
    for i in lists_dict:
        print(f"\t{i}\t{lists_dict[i]}")


if __name__ == "__main__":
    while True:
        print("The available lists are: ")
        print_lists()
        print("Enter a list number to plot it. "
              "\nEnter -1 to exit")
        value_inserted = input()
        command_inserted: Optional[int] = sim.get_integer(value_inserted)

        if command_inserted is None:
            print(f"You inserted {value_inserted}, expected an integer number")
        elif command_inserted == -1:
            exit()
        elif command_inserted in lists_dict.keys():
            list_name = lists_dict[command_inserted]
            print("You selected list: " + list_name)
            list: ListsNames = ListsNames[list_name]
            MoviesLists(list).plot()
        else:
            print(f"The number {command_inserted} does not correspond to a list")
