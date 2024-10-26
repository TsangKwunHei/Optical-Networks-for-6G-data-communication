# Highest Response Ratio Next Scheduling Algorithm (Non-Preemptive)

class Process:
    def __init__(self, no, at, bt):
        self.no = no       # Process Number
        self.at = at       # Arrival Time
        self.bt = bt       # Burst Time
        self.ct = 0        # Completion Time
        self.tat = 0       # Turnaround Time
        self.wt = 0        # Waiting Time
        self.completed = False

def read_process(i):
    print(f"\nProcess No: {i}")
    at = int(input("Enter Arrival Time: "))
    bt = int(input("Enter Burst Time: "))
    return Process(i, at, bt)

def hrrn_scheduling(processes, n):
    # Sort processes based on Arrival Time
    processes.sort(key=lambda x: x.at)
    
    remaining = n
    current_time = processes[0].at
    total_tat = 0
    total_wt = 0

    print("\nProcess\tAT\tBT\tCT\tTAT\tWT")
    
    while remaining > 0:
        highest_response_ratio = -9999
        selected_process = None

        for process in processes:
            if process.at <= current_time and not process.completed:
                response_ratio = (process.bt + (current_time - process.at)) / process.bt
                if response_ratio > highest_response_ratio:
                    highest_response_ratio = response_ratio
                    selected_process = process

        if selected_process:
            current_time += selected_process.bt
            selected_process.ct = current_time
            selected_process.tat = selected_process.ct - selected_process.at
            selected_process.wt = selected_process.tat - selected_process.bt
            selected_process.completed = True
            remaining -= 1

            total_tat += selected_process.tat
            total_wt += selected_process.wt

            print(f"P{selected_process.no}\t{selected_process.at}\t{selected_process.bt}\t"
                  f"{selected_process.ct}\t{selected_process.tat}\t{selected_process.wt}")
        else:
            # If no process is ready to execute, increment time
            current_time += 1

    avg_tat = total_tat / n
    avg_wt = total_wt / n

    print(f"\nAverage TurnAround Time = {avg_tat:.2f}")
    print(f"Average Waiting Time = {avg_wt:.2f}")

def main():
    print("<--Highest Response Ratio Next Scheduling Algorithm (Non-Preemptive)-->")
    n = int(input("Enter Number of Processes: "))
    processes = []

    for i in range(1, n + 1):
        processes.append(read_process(i))

    hrrn_scheduling(processes, n)

if __name__ == "__main__":
    main()