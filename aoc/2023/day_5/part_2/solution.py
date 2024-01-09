import sys
from pathlib import Path

class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __repr__(self):
        return str((self.start, self.end))

    def part_before(self, other):
        if self.start < other.start:
            return Range(self.start, min(self.end, other.start-1))

    def part_inside(self, other):
        if not self.end < other.start and not other.end < self.start:
            return Range(max(self.start, other.start), min(self.end, other.end))

    def part_after(self, other):
        if other.end < self.end:
            return Range(max(self.start, other.end+1), self.end)

    def shift(self, shift):
        return Range(self.start + shift, self.end + shift)

class Mapping:
    @staticmethod
    def from_section(section):
        mappings = []
        for to_range_start, from_range_start, length in section:
            from_range = Range(from_range_start, from_range_start+length-1)
            to_range = Range(to_range_start, to_range_start+length-1)
            mappings.append((from_range, to_range))
        mappings = sorted(mappings, key=lambda v: v[0].start)
        return Mapping(mappings)

    def __init__(self, mappings):
        self.mappings = mappings

    def __getitem__(self, mapped_range):
        result = []
        for from_range, to_range in self.mappings:
            part_before = mapped_range.part_before(from_range)
            part_inside = mapped_range.part_inside(from_range)
            part_after = mapped_range.part_after(from_range)

            if part_before:
                result.append(part_before)  # no mapping
            if part_inside:
                shift = to_range.start - from_range.start
                result.append(part_inside.shift(shift))  # mapping

            mapped_range = part_after
            if mapped_range is None:
                break
        if mapped_range:
            result.append(mapped_range)

        result = sorted(result, key=lambda v: v.start)
        return result

def solve(input: str):
    lines = input.strip().split('\n')
    seed_numbers = [int(s) for s in lines[0][len("seeds:"):].strip().split(' ')]
    seed_ranges = [Range(start, start+length-1) for start, length in zip(seed_numbers[::2], seed_numbers[1::2])]

    sections = input.strip().split('\n\n')[1:]
    sections = [
        [[int(n) for n in line.split(' ')] for line in s.split('\n')[1:]]
        for s in sections
    ]
    for section in sections:
        mapping = Mapping.from_section(section)
        section_results = []
        for seed_range in seed_ranges:
            section_results.extend(mapping[seed_range])
        seed_ranges = section_results
    return sorted(seed_ranges, key=lambda v: v.start)[0].start


if __name__ == "__main__":
    input = Path(sys.argv[1]).read_text()
    output = solve(input)
    print(output)
