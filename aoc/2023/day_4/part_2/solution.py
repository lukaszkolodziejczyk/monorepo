import sys
import heapq
from pathlib import Path


def solve(input: str):
    lines = input.strip().split('\n')

    card_strength = {}
    heap = []
    for line in lines:
        card, line = line.split(':')
        card = int(card[len('Card'):])
        winning, chosen = line.split(' | ')
        winning = [n for n in winning.split(' ') if n]
        chosen = [n for n in chosen.split(' ') if n]
        common = set(winning) & set(chosen)
        strength = len(common)
        card_strength[card] = strength
        heapq.heappush(heap, (card, strength))

    n_cards = 0
    while heap:
        card, strength = heapq.heappop(heap)
        for i in range(1, strength + 1):
            heapq.heappush(heap, (card + i, card_strength[card + i]))
        n_cards += 1
    return n_cards


if __name__ == "__main__":
    input = Path(sys.argv[1]).read_text()
    output = solve(input)
    print(output)
