import sys
import threading
import math
import heapq
from collections import deque
import os
from pathlib import Path
import json  # Import JSON for event logging

# Set DEBUG to True to enable debug statements, False to disable
DEBUG = True

def debug_print(*args):
    if DEBUG:
        print("[DEBUG]", *args, file=sys.stderr)

def main():
    import sys

    # Determine the directory where the script is located
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    debug_print(f"Script directory: {script_dir}")

    # Handle input source based on DEBUG mode
    if DEBUG:
        input_file = script_dir / 'input.txt'
        debug_print(f"Looking for input file at: {input_file}")
        if input_file.is_file():
            try:
                with open(input_file, 'r') as f:
                    data = f.read().split()
                debug_print("Debug mode enabled. Reading input from 'input.txt'.")
            except Exception as e:
                debug_print(f"Error reading 'input.txt': {e}")
                debug_print("Falling back to standard input.")
                data = sys.stdin.read().split()
        else:
            debug_print(f"Debug mode enabled but '{input_file}' not found. Falling back to standard input.")
            data = sys.stdin.read().split()
    else:
        data = sys.stdin.read().split()

    idx = 0

    # Number of slices and PortBW
    if idx >= len(data):
        debug_print("Error: No data found for number of slices and PortBW.")
        return
    try:
        n = int(data[idx]); idx += 1
        PortBW_Gbps = float(data[idx]); idx += 1  # in Gbps
        PortBW = PortBW_Gbps   
        debug_print(f"Number of slices: {n}, PortBW: {PortBW} Gbps")
    except (IndexError, ValueError) as e:
        debug_print(f"Error parsing number of slices and PortBW: {e}")
        return

    slices = []
    all_packets = []
    event_log = []  # Initialize event log

    for slice_id in range(n):
        if idx + 2 >= len(data):
            debug_print(f"Error: Insufficient data for slice {slice_id}.")
            return
        try:
            m_i = int(data[idx]); idx += 1
            SliceBW_i = float(data[idx]); idx += 1  # in Gbps
            UBD_i = int(data[idx]); idx += 1  # in ns
            debug_print(f"Slice {slice_id}: m_i={m_i}, SliceBW_i={SliceBW_i} Gbps, UBD_i={UBD_i} ns")
        except (IndexError, ValueError) as e:
            debug_print(f"Error parsing slice {slice_id} information: {e}")
            return
        
        packets = []
        for pkt_id in range(m_i):
            if idx + 1 >= len(data):
                debug_print(f"Error: Insufficient data for packet {pkt_id} in slice {slice_id}.")
                return
            try:
                ts = int(data[idx]); idx += 1  # arrival time in ns
                pkt_size = int(data[idx]); idx += 1  # in bits
                deadline = ts + UBD_i
                packet = {
                    'ts': ts,
                    'pkt_size': pkt_size,
                    'pkt_id': pkt_id,
                    'slice_id': slice_id,
                    'deadline': deadline
                }
                packets.append(packet)
                all_packets.append(packet)
                debug_print(f"  Packet {pkt_id}: ts={ts}, pkt_size={pkt_size} bits, deadline={deadline} ns")
            except (IndexError, ValueError) as e:
                debug_print(f"Error parsing packet {pkt_id} in slice {slice_id}: {e}")
                return
        
        slice_info = {
            'UBD_i': UBD_i,
            'SliceBW_i': SliceBW_i,
            'slice_id': slice_id,
            'packets': deque(sorted(packets, key=lambda x: x['ts'])),  # Ensure packets are ordered
            'total_bits_sent': 0,
            'first_ts': packets[0]['ts'] if packets else 0,
            'last_te': 0,
            'max_delay': 0,
            'scheduled_te_list': []  # To track departure times for Constraint 3
        }
        slices.append(slice_info)
        debug_print(f"Initialized slice {slice_id} with {len(packets)} packets.")

    # Sort all packets by arrival time
    all_packets.sort(key=lambda pkt: pkt['ts'])
    total_packets = len(all_packets)
    debug_print(f"Total number of packets to schedule: {total_packets}")

    # Scheduled packets list
    scheduled_packets = []

    # Initialize variables
    current_time = 0
    port_available_time = 0
    packet_idx = 0
    heap = []

    # To track the departure time of the previous packet for Constraint 1
    previous_te = None

    # Priority function: (earliest deadline, highest SliceBW, earliest slice_id)
    def get_priority(pkt):
        return (pkt['deadline'], -slices[pkt['slice_id']]['SliceBW_i'], slices[pkt['slice_id']]['slice_id'])

    # Main scheduling loop
    while packet_idx < total_packets or heap:
        # Add all packets that have arrived up to current_time to the heap
        while packet_idx < total_packets and all_packets[packet_idx]['ts'] <= current_time:
            pkt = all_packets[packet_idx]
            heapq.heappush(heap, (get_priority(pkt), pkt))
            debug_print(f"Packet {pkt['pkt_id']} from Slice {pkt['slice_id']} added to heap with priority {get_priority(pkt)}.")
            packet_idx += 1

        if not heap:
            # No packets to schedule, advance time
            if packet_idx < total_packets:
                next_ts = all_packets[packet_idx]['ts']
                new_time = max(current_time, next_ts, port_available_time)
                debug_print(f"No packets in heap. Advancing current_time to {new_time} ns.")
                current_time = new_time
            else:
                break
            continue

        # Pop the packet with the highest priority
        _, pkt = heapq.heappop(heap)
        slice_id = pkt['slice_id']
        s = slices[slice_id]
        debug_print(f"Scheduling Packet {pkt['pkt_id']} from Slice {slice_id} at current_time {current_time} ns.")

        # Ensure packets are scheduled in order within the slice
        if s['packets'] and s['packets'][0]['pkt_id'] != pkt['pkt_id']:
            # Not the next packet in slice, skip and re-add to heap
            heapq.heappush(heap, (get_priority(pkt), pkt))
            debug_print(f"Packet {pkt['pkt_id']} from Slice {slice_id} is not next in queue. Re-added to heap.")
            # Advance current_time to the earliest possible next packet
            if s['packets']:
                next_pkt = s['packets'][0]
                new_time = max(current_time, next_pkt['ts'], port_available_time)
                debug_print(f"Advancing current_time to {new_time} ns based on next packet in Slice {slice_id}.")
                current_time = new_time
            continue

        # Pop the next packet from the slice's queue
        if s['packets']:
            next_packet = s['packets'].popleft()
            debug_print(f"Popped Packet {next_packet['pkt_id']} from Slice {slice_id} queue.")
        else:
            next_packet = None
            debug_print(f"No packets left in Slice {slice_id} queue.")

        # Determine earliest possible departure time
        te_candidate = max(current_time, pkt['ts'], s['last_te'], port_available_time)
        te = math.ceil(te_candidate)
        debug_print(f"Calculated te_candidate: {te_candidate} ns, final te: {te} ns.")

        # Calculate transmission time
        transmission_time = math.ceil(pkt['pkt_size'] / PortBW)
        debug_print(f"Transmission time for Packet {pkt['pkt_id']} from Slice {slice_id}: {transmission_time} ns.")

        # Constraint 1: Port Bandwidth Constraint
        if previous_te is not None:
            required_interval = pkt['pkt_size'] / PortBW
            actual_interval = te - previous_te
            debug_print(f"Constraint 1 - PortBW:")
            debug_print(f"  Previous te: {previous_te} ns")
            debug_print(f"  Current te: {te} ns")
            debug_print(f"  Required interval: {required_interval} ns")
            debug_print(f"  Actual interval: {actual_interval} ns")
            if actual_interval < required_interval:
                debug_print(f"  Constraint 1 Violated: {actual_interval} < {required_interval}")
            else:
                debug_print(f"  Constraint 1 Satisfied: {actual_interval} >= {required_interval}")
        else:
            debug_print("Constraint 1: First packet being scheduled. No previous te to compare.")

        # Update previous_te for the next packet
        previous_te = te

        # Update port availability
        port_available_time = te + transmission_time
        debug_print(f"Port available at {port_available_time} ns after transmitting Packet {pkt['pkt_id']}.")

        # Constraint 3: Order and Departure Time within Slice
        # Ensure te >= ts
        debug_print(f"Constraint 3 - Packet Departure Time:")
        debug_print(f"  te (departure time): {te} ns")
        debug_print(f"  ts (arrival time): {pkt['ts']} ns")
        if te < pkt['ts']:
            debug_print(f"  Constraint 3 Violated: te ({te}) < ts ({pkt['ts']})")
        else:
            debug_print(f"  Constraint 3 Satisfied: te ({te}) >= ts ({pkt['ts']})")

        # Ensure order within the slice
        if s['scheduled_te_list']:
            last_te_in_slice = s['scheduled_te_list'][-1]
            debug_print(f"  Last te in Slice {slice_id}: {last_te_in_slice} ns")
            if te < last_te_in_slice:
                debug_print(f"  Constraint 3 Violated: te ({te}) < last_te_in_slice ({last_te_in_slice})")
            else:
                debug_print(f"  Constraint 3 Satisfied: te ({te}) >= last_te_in_slice ({last_te_in_slice})")
        else:
            debug_print(f"  No previous te in Slice {slice_id} to compare for order.")

        # Record the departure time in the slice's schedule
        s['scheduled_te_list'].append(te)

        # Calculate transmission time
        # (Assuming PortBW is in Gbps and pkt_size is in bits, transmission_time should be in ns
        # So, transmission_time = pkt_size / (PortBW * 1e9) * 1e9 = pkt_size / PortBW
        # Which is already calculated above.

        # Update slice's last_te
        s['last_te'] = te
        debug_print(f"Slice {slice_id} last_te updated to {te} ns.")

        # Update slice's metrics
        s['total_bits_sent'] += pkt['pkt_size']
        delay = te - pkt['ts']
        debug_print(f"Packet {pkt['pkt_id']} delay: {delay} ns.")
        if delay > s['max_delay']:
            debug_print(f"Updating Slice {slice_id} max_delay from {s['max_delay']} to {delay} ns.")
            s['max_delay'] = delay

        # Record the scheduled packet
        scheduled_packets.append({
            'te': te,
            'slice_id': slice_id,
            'pkt_id': pkt['pkt_id'],
            'ts': pkt['ts'],
            'departure_time': te,
            'delay': delay
        })
        debug_print(f"Scheduled Packet {pkt['pkt_id']} from Slice {slice_id} at te={te} ns.")

        # Log the event
        event = {
            'time': te,
            'action': 'departure',
            'packet_id': pkt['pkt_id'],
            'slice_id': slice_id,
            'ts': pkt['ts'],
            'departure_time': te,
            'delay': delay
        }
        event_log.append(event)

        # Advance current time
        current_time = te
        debug_print(f"Advancing current_time to {current_time} ns.")

    # After scheduling all packets, check Constraint 2 for each slice
    debug_print("\nConstraint 2 - Output Bandwidth Constraint for Each Slice:")
    for s in slices:
        if s['first_ts'] == 0 and not s['scheduled_te_list']:
            # No packets were scheduled for this slice
            debug_print(f"Slice {s['slice_id']} has no scheduled packets.")
            continue
        interval = s['last_te'] - s['first_ts']
        if interval == 0:
            computed_SliceBW = float('inf')  # Avoid division by zero
            debug_print(f"Slice {s['slice_id']} interval is 0 ns. Setting computed_SliceBW to infinity.")
        else:
            computed_SliceBW = s['total_bits_sent'] / interval  # bits/ns = Gbps (since 1 bit/ns = 1 Gbps)
        required_SliceBW = s['SliceBW_i'] * 0.95
        debug_print(f"Slice {s['slice_id']}:")
        debug_print(f"  Total Packet Size: {s['total_bits_sent']} bits")
        debug_print(f"  Interval: {interval} ns")
        debug_print(f"  Computed SliceBW: {computed_SliceBW} Gbps")
        debug_print(f"  Required SliceBW (95% of SliceBW_i): {required_SliceBW} Gbps")
        if computed_SliceBW >= required_SliceBW:
            debug_print(f"  Constraint 2 Satisfied for Slice {s['slice_id']}: {computed_SliceBW} >= {required_SliceBW}")
        else:
            debug_print(f"  Constraint 2 Violated for Slice {s['slice_id']}: {computed_SliceBW} < {required_SliceBW}")

    # Calculate the score based on the problem's scoring formula
    fi_sum = 0
    max_delay = 0
    for s in slices:
        fi = 1 if s['max_delay'] <= s['UBD_i'] else 0
        fi_sum += fi / n
        if s['max_delay'] > max_delay:
            max_delay = s['max_delay']
    if max_delay > 0:
        score = fi_sum + (10000 / max_delay)
    else:
        # Handle division by zero if max_delay is 0
        score = fi_sum + 10000
        debug_print(f"Warning: max_delay is 0. Setting second term of score to 10000 to avoid division by zero.")

    debug_print(f"\nScore Calculation:")
    debug_print(f"  Sum(fi / n): {fi_sum}")
    debug_print(f"  Max Delay: {max_delay} ns")
    debug_print(f"  Score: {score}")

    # Output generation
    K = len(scheduled_packets)
    print(K)
    output = []
    for pkt in scheduled_packets:
        output.append(f"{pkt['departure_time']} {pkt['slice_id']} {pkt['pkt_id']}")
    print(' '.join(output))
    debug_print(f"Total scheduled packets: {K}")
    debug_print(f"Scheduling sequence: {' '.join(output)}")
    debug_print("Script completed successfully.")

    # Save the event log to a JSON file for visualization
    with open(script_dir / 'event_log.json', 'w') as f:
        json.dump(event_log, f, indent=4)
    debug_print("Event log saved to 'event_log.json'.")

if __name__ == "__main__":
    try:
        threading.Thread(target=main).start()
    except KeyboardInterrupt:
        debug_print("Script interrupted by user.")
