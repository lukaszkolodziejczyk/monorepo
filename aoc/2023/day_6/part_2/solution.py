import sys
from pathlib import Path

def simulate(velocity_build_time, race_time):
    velocity = velocity_build_time
    move_time = race_time - velocity_build_time
    return velocity * move_time


def solve(input: str):
    lines = input.strip().split('\n')
    time_line, distance_line = lines
    race_time = int(''.join([x for x in time_line[len("Time:"):].strip().split(' ') if x]))
    record_distance = int(''.join([x for x in distance_line[len("Distance:"):].strip().split(' ') if x]))
    wins = 0
    queue = [(0, race_time)]
    while queue:
        time_left, time_right = queue.pop()
        if time_left > time_right:
            continue
        time_mid = time_left + (time_right - time_left) // 2
        distance = simulate(time_mid, race_time)
        queue.append((time_left, time_mid - 1))
        queue.append((time_mid + 1, time_right))
        if distance > record_distance:
            wins += 1
    return wins


if __name__ == "__main__":
    input = Path(sys.argv[1]).read_text()
    output = solve(input)
    print(output)
