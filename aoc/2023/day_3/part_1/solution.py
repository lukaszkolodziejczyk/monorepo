import sys
import string
from pathlib import Path


class Number:
    def __init__(self, number, neighbours):
        self.number = number
        self.neighbours = neighbours

    def __repr__(self):
        return f"{self.number} {self.neighbours}"


def solve(input: str):
    lines = input.strip().split('\n')
    lines = ['.' * len(lines[0])] + lines
    lines = lines + ['.' * len(lines[0])]
    lines = ['.' + l + '.' for l in lines]

    numbers = []
    for row in range(1, len(lines)):
        line = lines[row]
        curr_num = False
        number = []
        neighbours = []
        for column in range(1, len(line)):
            char = line[column]
            prev_num = curr_num
            curr_num = char in list(string.digits)
            if curr_num:
                number.append(char)
                neighbours.append(lines[row - 1][column])  # one up
                neighbours.append(lines[row + 1][column])  # one down
            if not prev_num and curr_num:
                neighbours.append(lines[row - 1][column - 1])  # one up-left
                neighbours.append(lines[row][column - 1])  # one left
                neighbours.append(lines[row + 1][column - 1])  # one down-left
            if prev_num and not curr_num:
                neighbours.append(lines[row - 1][column])  # one up-right
                neighbours.append(lines[row][column])  # one right
                neighbours.append(lines[row + 1][column])  # one down-right
                numbers.append(Number(int(''.join(number)), neighbours))
                number = []
                neighbours = []

    total = 0
    for number in numbers:
        if [e for e in number.neighbours if e != '.']:
            total += number.number
    return total


if __name__ == "__main__":
    input = Path(sys.argv[1]).read_text()
    output = solve(input)
    print(output)
