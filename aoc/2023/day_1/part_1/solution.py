import sys
from pathlib import Path


def solve(input: str):
    lines = input.strip().split('\n')
    numbers = [str(n) for n in range(10)]
    collected = []
    for line in lines:
        left_num = None
        for c in line:
            if c in numbers:
                left_num = c
                break
        right_num = None
        for c in reversed(line):
            if c in numbers:
                right_num = c
                break
        assert left_num is not None and right_num is not None
        collected.append(int(left_num + right_num))
    return sum(collected)


if __name__ == "__main__":
    input = Path(sys.argv[1]).read_text()
    output = solve(input)
    print(output)
