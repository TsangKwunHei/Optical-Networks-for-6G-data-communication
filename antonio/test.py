# pyright: reportUnusedCallResult=false, reportAny=false
import subprocess
from typing import Any

# Initial input data
initial_input = """5 6
1 1 1 1 1
1 2
2 5
1 4
4 5
1 3
3 5
2
1 5 2 1 20 1
1 2
1 5 2 21 40 1
1 2""".encode()

# Start the process
process = subprocess.Popen(
    ["python", "main.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
)

# Send initial input
if process.stdin is None or process.stdout is None:
    raise Exception("std is None")
process.stdin.write(initial_input + b"\n")
process.stdin.flush()


def readline():
    if process.stdout is None:
        raise Exception("stdout is None")
    return process.stdout.readline().decode().strip()


def writeline(s: Any):
    if process.stdin is None:
        raise Exception("stdin is None")
    process.stdin.write(f"{str(s)}\n".encode())
    process.stdin.flush()


test_scenarios = int(readline())
n_edge_failures = int(readline())
edge_failures = []
if n_edge_failures != 0:
    edge_failures = [int(n) for n in readline().split(" ")]
if len(edge_failures) != n_edge_failures:
    raise Exception("Invalid number of edge failures")

expected_meta = [
    ["1 2", "2 2"],
    ["1 2", "2 2"],
]
expected_services = [
    ["5 1 20 6 1 20", "5 21 40 6 21 40"],
    ["3 1 20 4 1 20", "3 21 40 4 21 40"],
]

writeline(test_scenarios)
for n, e in enumerate(edge_failures):
    writeline(e)
    num_success_replans = int(readline())
    print(num_success_replans)
    assert num_success_replans == 2
    for j in range(num_success_replans):
        assert readline() == expected_meta[n][j]
        assert readline() == expected_services[n][j]
        # print(readline())
        # print(readline())
    process.stdin.flush()

writeline(-1)
# Clean up
process.stdin.close()
process.stdout.close()
process.wait()
print("Done")
