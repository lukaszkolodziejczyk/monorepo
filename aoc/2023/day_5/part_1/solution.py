import sys
from pathlib import Path

def forward(seed, ranges):
    for r in ranges:
        destination_start, source_start, length = r
        source_end = source_start + length - 1
        if source_start <= seed <= source_end:
            shift = seed - source_start
            return destination_start + shift
    return seed

def solve(input: str):
    lines = input.strip().split('\n')
    seeds = [int(s) for s in lines[0][len("seeds:"):].strip().split(' ')]
    sections = input.strip().split('\n\n')[1:]
    sections = [
        [[int(n) for n in line.split(' ')] for line in s.split('\n')[1:]]
        for s in sections
    ]
    results = []
    for seed in seeds:
        for section in sections:
            seed = forward(seed, section)
        results.append(seed)
    return min(results)


if __name__ == "__main__":
    input = Path(sys.argv[1]).read_text()
    output = solve(input)
    print(output)
