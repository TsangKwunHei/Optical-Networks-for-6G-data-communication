# pyright: basic
import subprocess
import threading


def pipe_output(process, other_process, prefix):
    for line in iter(process.stdout.readline, b""):
        # print(f"{prefix}: {line.decode().strip()}")
        other_process.stdin.write(line)
        other_process.stdin.flush()
    process.stdout.close()


def print_stderr(process, prefix):
    for line in iter(process.stderr.readline, b""):
        print(f"{prefix} stderr: {line.decode().strip()}")
    process.stderr.close()


# Start the two Python scripts
process1 = subprocess.Popen(
    ["python3", "main.py"],
    stdout=subprocess.PIPE,
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

process2 = subprocess.Popen(
    ["python3", "evaluator.py"],
    stdout=subprocess.PIPE,
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Create threads to handle the output of each process

# Create threads to handle the output and stderr of each process
thread1_out = threading.Thread(target=pipe_output, args=(process1, process2, "Main"))
thread1_err = threading.Thread(target=print_stderr, args=(process1, "Main"))
thread2_out = threading.Thread(
    target=pipe_output, args=(process2, process1, "Evaluator")
)
thread2_err = threading.Thread(target=print_stderr, args=(process2, "Evaluator"))


# Start the threads
thread1_out.start()
thread1_err.start()
thread2_out.start()
thread2_err.start()

# Wait for both processes to complete
process1.wait()
process2.wait()

if process1.stdin is None or process2.stdin is None:
    raise Exception("Process stdin is None")
# Close the pipes
process1.stdin.close()
process2.stdin.close()


# Wait for the threads to finish
thread1_out.join()
thread1_err.join()
thread2_out.join()
thread2_err.join()
