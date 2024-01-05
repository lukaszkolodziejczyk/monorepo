import sys
from pathlib import Path


def solve(input: str):
    lines = input.strip().split('\n')
    red, green, blue = 12, 13, 14
    possible_games = []
    for line in lines:
        game_x, subsets = line.split(":")
        game_id = int(game_x[len("Game "):])
        subsets = subsets.split(";")
        subsets = [
            [

                [num_cube for num_cube in cubes.strip().split(' ')]
                for cubes in subset.strip().split(',')
            ]
            for subset in subsets
        ]
        is_possible = True
        for subset in subsets:
            for number, cube in subset:
                if any([
                    cube == "red" and int(number) > red,
                    cube == "green" and int(number) > green,
                    cube == "blue" and int(number) > blue
                ]):
                    is_possible = False
        if is_possible:
            possible_games.append(game_id)
    return sum(possible_games)


if __name__ == "__main__":
    input = Path(sys.argv[1]).read_text()
    output = solve(input)
    print(output)
