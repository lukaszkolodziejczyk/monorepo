import sys
from pathlib import Path


def solve(input: str):
    lines = input.strip().split('\n')
    total_score = 0
    for line in lines:
        line = line.split(': ')[1]
        winning, chosen = line.split(' | ')
        winning = [n for n in winning.split(' ') if n]
        chosen = [n for n in chosen.split(' ') if n]
        common = set(winning) & set(chosen)
        score = 1 << (len(common) - 1) if common else 0
        total_score += score
    return total_score


if __name__ == "__main__":
    input = Path(sys.argv[1]).read_text()
    output = solve(input)
    print(output)
