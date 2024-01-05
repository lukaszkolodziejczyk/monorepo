import sys
from pathlib import Path


def solve(input: str):
    lines = input.strip().split('\n')
    numbers = {str(n): str(n) for n in range(10)} | {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9"
    }
    collected = []
    for line in lines:
        line_nums = []
        partial_line = ""
        for c in line:
            partial_line += c
            for n in numbers:
                if partial_line.endswith(n):
                    line_nums.append(n)
                    break
        left_num, right_num = numbers[line_nums[0]], numbers[line_nums[-1]]
        assert left_num is not None and right_num is not None
        collected.append(int(left_num + right_num))
    return sum(collected)


if __name__ == "__main__":
    input = Path(sys.argv[1]).read_text()
    output = solve(input)
    print(output)
