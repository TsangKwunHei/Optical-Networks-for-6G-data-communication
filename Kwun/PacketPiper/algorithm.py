#@title Implement the Final Scheduling Algorithm
import sys
import threading
import math
import heapq
from collections import deque
import time  # For checking runtime
def main ():
    data = sys.stdin.read()
    final_scheduling_algorithm(data)

def final_scheduling_algorithm(input_data, weights_file='policy_weights.npz'):
    import math
    import heapq
    from collections import deque
    import numpy as np
    import time

    start_time = time.time()

    # Parse input data
    data = input_data.split()
    index = 0

    num_slices = int(data[index])
    index += 1
    port_bw_gbps = float(data[index])
    index += 1
    port_bw = port_bw_gbps

    slices = []
    all_packets = []

    for slice_id in range(num_slices):
        num_slice_packets = int(data[index])
        index += 1
        slice_bw = float(data[index])
        index += 1
        slice_delay_tolerance = int(data[index])
        index += 1

        packets = []
        for packet_id in range(num_slice_packets):
            arrival_time_ns = int(data[index])
            index += 1
            packet_size = int(data[index])
            index += 1
            deadline = arrival_time_ns + slice_delay_tolerance

            packet = {
                'ts': arrival_time_ns,
                'pkt_size': packet_size,
                'pkt_id': packet_id,
                'slice_id': slice_id,
                'deadline': deadline
            }
            packets.append(packet)
            all_packets.append(packet)

        sorted_packets = deque(sorted(packets, key=lambda pkt: pkt['ts']))

        slice_info = {
            'ubd': slice_delay_tolerance,
            'slice_bw': slice_bw,
            'slice_id': slice_id,
            'packets': sorted_packets,
            'total_bits_sent': 0,
            'first_ts': packets[0]['ts'] if packets else 0,
            'last_te': 0,
            'max_delay': 0,
            'scheduled_packets': []
        }
        slices.append(slice_info)

    all_packets.sort(key=lambda pkt: pkt['ts'])
    total_packets = len(all_packets)

    scheduled_packets = []

    current_time = 0
    port_available_time = 0
    packet_index = 0
    heap = []

    # Load policy weights
    policy_weights = np.load(weights_file, allow_pickle=True)
    w1 = policy_weights['w1']
    b1 = policy_weights['b1']
    w2 = policy_weights['w2']
    b2 = policy_weights['b2']
    w3 = policy_weights['w3']
    b3 = policy_weights['b3']

    def relu(x):
        return np.maximum(0, x)

    def get_priority(packet, current_time, port_available_time):
        # Feature vector: is_available, is_scheduled, deadline, time_until_deadline
        is_available = 1 if packet['ts'] <= current_time else 0
        is_scheduled = 1 if packet['scheduled'] else 0
        deadline_normalized = packet['deadline'] / 100000
        time_until_deadline = (packet['deadline'] - current_time) / 100000
        features = np.array([is_available, is_scheduled, deadline_normalized, time_until_deadline, 
                             current_time / 100000, port_available_time / 100000], dtype=np.float32)

        # Forward pass through the network
        x = np.dot(features, w1) + b1
        x = relu(x)
        x = np.dot(x, w2) + b2
        x = relu(x)
        x = np.dot(x, w3) + b3
        priority = x[0]  # Assuming single output neuron

        return priority

    while packet_index < total_packets or heap:
        # Update availability of packets based on current_time
        while packet_index < total_packets and all_packets[packet_index]['ts'] <= current_time:
            packet = all_packets[packet_index]
            priority = get_priority(packet, current_time, port_available_time)
            heapq.heappush(heap, (-priority, packet))  # Max-heap using negative priority
            packet_index += 1

        if not heap:
            # No packets to schedule, advance current_time
            if packet_index < total_packets:
                next_packet_ts = all_packets[packet_index]['ts']
                current_time = max(current_time, next_packet_ts, port_available_time)
            else:
                break
            continue

        # Pop the packet with the highest priority
        _, packet = heapq.heappop(heap)
        slice_id = packet['slice_id']
        slice_info = slices[slice_id]

        # Ensure packets are scheduled in the order they arrived within the same slice
        if slice_info['packets'] and slice_info['packets'][0]['pkt_id'] != packet['pkt_id']:
            heapq.heappush(heap, (-get_priority(packet, current_time, port_available_time), packet))
            if slice_info['packets']:
                next_packet = slice_info['packets'][0]
                current_time = max(next_packet['ts'], port_available_time)
            continue

        # Remove packet from slice's queue
        if slice_info['packets']:
            next_packet = slice_info['packets'].popleft()
        else:
            next_packet = None

        # Determine departure time
        te_candidate = max(
            current_time,
            packet['ts'],
            slice_info['last_te'],
            port_available_time
        )
        te = math.ceil(te_candidate)

        # Transmission time
        transmission_time = math.ceil(packet['pkt_size'] / port_bw)

        # Update port availability
        port_available_time = te + transmission_time

        # Update slice's last departure time
        slice_info['last_te'] = te

        # Update slice metrics
        slice_info['total_bits_sent'] += packet['pkt_size']
        delay = te - packet['ts']
        if delay > slice_info['max_delay']:
            slice_info['max_delay'] = delay

        # Mark packet as scheduled
        packet['scheduled'] = True

        # Record scheduled packet
        scheduled_packets.append((te, slice_id, packet['pkt_id'], packet['pkt_size'], packet['ts']))
        slice_info['scheduled_packets'].append((te, packet['pkt_id'], packet['pkt_size'], packet['ts']))

        # Advance current_time
        current_time = te

    # Calculate the score
    fi_sum = 0
    max_delay = 0
    for slice_info in slices:
        fi = 1 if slice_info['max_delay'] <= slice_info['ubd'] else 0
        fi_sum += fi / num_slices
        if slice_info['max_delay'] > max_delay:
            max_delay = slice_info['max_delay']

    if max_delay > 0:
        score = fi_sum + (10000 / max_delay)
    else:
        score = fi_sum + 10000

    # Output generation
    K = len(scheduled_packets)
    print(K)
    output = []
    for te, slice_id, pkt_id, _, _ in scheduled_packets:
        output.append(f"{te} {slice_id} {pkt_id}")
    print(' '.join(output))

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Ensure runtime is within constraints
    if elapsed_time > 120:
        raise Exception("Constraint violated: Runtime exceeds 2 minutes.")
    

if __name__ == "__main__":
    # Start the main function in a separate thread to avoid potential recursion limits
    threading.Thread(target=main).start()
