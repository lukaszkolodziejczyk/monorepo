import sys
from pathlib import Path


def solve(input: str):
    lines = input.strip().split('\n')
    minimum_powers = []
    for line in lines:
        _, subsets = line.split(":")
        subsets = subsets.split(";")
        subsets = [
            [

                [num_cube for num_cube in cubes.strip().split(' ')]
                for cubes in subset.strip().split(',')
            ]
            for subset in subsets
        ]
        cubes = {"red": -1, "green": -1, "blue": -1}
        for subset in subsets:
            for number, cube in subset:
                cubes[cube] = max(cubes[cube], int(number))
        minimum_powers.append(cubes["red"] * cubes["green"] * cubes["blue"])
    return sum(minimum_powers)


if __name__ == "__main__":
    input = Path(sys.argv[1]).read_text()
    output = solve(input)
    print(output)
