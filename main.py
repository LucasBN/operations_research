import numpy as np


def read_dataset(filepath):
    x = []
    for line in open(filepath):
        if line.strip() != "":
            row = line.strip().split(" ")
            x.append(list(map(float, row)))

    return np.array(x)


def optimise(data):
    # Iteratively apply the simple algorithm until termination
    while data[0][1:-1].max() > 0:
        # Choose NBV to enter the basis
        column = np.argmax(data[0][1:]) + 1

        # Choose BV to leave the basis
        with np.errstate(divide="ignore"):
            ratios = data[:, -1] / data[:, column]
        row = np.argmin(np.where(ratios > 0, ratios, np.inf))

        # Normalise the row corresponding to BV leaving the basis
        data[row] = data[row] / data[row][column]

        # Pivot to adjacent vertex with a lower cost
        for i, d_row in enumerate(data):
            if i != row:
                data[i] = d_row - (data[row] * d_row[column])

    # Determine which indicies represent the basic variables
    basic_variables = []
    for i, r in enumerate(data.T[:-1]):
        if np.count_nonzero(r == 1) == 1 and np.count_nonzero(r == 0) == len(r) - 1:
            basic_variables.append(i)

    # Determine solution from data
    solution = np.zeros((len(data[0]) - 1))
    for i, bv in enumerate(basic_variables):
        solution[bv] = data[:, -1][i]

    return tuple(solution)


x = read_dataset("data.txt")
solution = optimise(x)

print(solution)
