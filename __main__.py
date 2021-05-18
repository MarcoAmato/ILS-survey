import src.similarities_util as sim

if __name__ == "__main__":
    while True:
        print("ILS creator. \n\tEnter 1 to create lists by entering the ids of top100 popularity movies manually"
              "\n\tEnter 2 to create lists from the similarities (as reported in the json files) of a random movie "
              "in the "
              "top100")
        value_inserted = input()
        command_inserted: int = -1
        try:
            command_inserted = int(value_inserted)
        except ValueError:
            print(f"You inserted {value_inserted}, expected an integer command")
        if command_inserted == 1:
            sim.print_ILS_from_ids()
        elif command_inserted == 2:
            sim.print_similar_movies_ILS()
        else:
            print(f"Command not valid {command_inserted}")
