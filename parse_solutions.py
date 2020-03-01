from pickle import dump, load


def get_solutions(filename = "data/solutions.pickle"):
    with open(filename, "rb") as f:
        return load(f)


if __name__ == "__main__":

    soldict = {}

    with open("data/solutions.txt") as f:
        for line in f:
            if line.startswith('#') or not line:
                continue

            line = line.strip().split(' : ')
            if len(line) < 2:
                continue

            soldict[line[0]] = int(line[1])

    with open("data/solutions.pickle", "wb") as f:
        dump(soldict, f)

