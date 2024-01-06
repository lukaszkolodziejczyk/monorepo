import sys
import string
from pathlib import Path


class Number:
    def __init__(self, number):
        self.number = number

    def __repr__(self):
        return f"{self.number}"


def solve(input: str):
    lines = input.strip().split('\n')
    lines = ['.' * len(lines[0])] + lines
    lines = lines + ['.' * len(lines[0])]
    lines = ['.' + l + '.' for l in lines]

    numbers = []
    coord_number = {}
    for row in range(1, len(lines)):
        line = lines[row]
        curr_num = False
        number = []
        coords = []
        for column in range(1, len(line)):
            char = line[column]
            prev_num = curr_num
            curr_num = char in list(string.digits)
            if curr_num:
                number.append(char)
                coords.append((row, column))
            if prev_num and not curr_num:
                n = Number(int(''.join(number)))
                numbers.append(n)
                for coord in coords:
                    coord_number[coord] = n
                number = []
                coords = []

    total = 0
    for row in range(1, len(lines)):
        for column in range(1, len(lines[row])):
            char = lines[row][column]
            if char == '*':
                numbers = set()
                for coord in [
                    (row-1, column-1), (row-1, column), (row-1, column+1),
                    (row, column-1)  ,                  (row, column+1)  ,
                    (row+1, column-1), (row+1, column), (row+1, column+1),
                ]:
                    if coord in coord_number and coord_number[coord] not in numbers:
                        numbers.add(coord_number[coord])
                if len(numbers) >= 2:
                    gear_ratio = numbers.pop().number * numbers.pop().number
                    total += gear_ratio
    return total


if __name__ == "__main__":
    input = Path(sys.argv[1]).read_text()
    output = solve(input)
    print(output)
