import sys
from collections import Counter
from pathlib import Path

def solve(input: str):
    lines = input.strip().split('\n')
    hands = [l.split(' ') for l in lines]
    hands = [(hand.strip(), int(bid)) for hand, bid in hands]
    strength = {
        c: i
        for i, c in enumerate(reversed(['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']))
    }
    type_hands = {
        "five_of_kind": [],
        "four_of_kind": [],
        "full_house": [],
        "three_of_kind": [],
        "two_pair": [],
        "one_pair": [],
        "high_card": []
    }
    counts_type = {
        (5,): "five_of_kind",
        (1, 4): "four_of_kind",
        (2, 3): "full_house",
        (1, 1, 3): "three_of_kind",
        (1, 2, 2): "two_pair",
        (1, 1, 1, 2): "one_pair",
        (1, 1, 1, 1, 1): "high_card"
    }
    hands = sorted(hands, key=lambda x: [strength[c] for c in x[0]])
    for hand, bid in hands:
        counter = Counter(hand)
        type = counts_type[tuple(sorted(counter.values()))]
        type_hands[type].append((hand, bid))
    hands = [(hand, bid) for hands in reversed(type_hands.values()) for hand, bid in hands]
    total = sum([rank * bid for rank, (_, bid) in enumerate(hands, start=1)])
    return total


if __name__ == "__main__":
    input = Path(sys.argv[1]).read_text()
    output = solve(input)
    print(output)
