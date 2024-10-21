import sys
import threading
import math
import heapq
from collections import deque
import time  # For checking runtime

def main():
    """
    Main function to execute the refined packet scheduling algorithm.
    
    This function reads input data, processes packet scheduling across multiple slices,
    and outputs the scheduling sequence along with relevant metrics.
    
    The scheduling aims to minimize the maximum packet delay across all slices.
    """
    # Start time for runtime constraint checking (Constraint 2)
    start_time = time.time()

    # Read all input data from standard input and split into tokens
    data = sys.stdin.read().split()
    index = 0  # Current position in the input data

    # Parse the number of slices and Port Bandwidth (Gbps)
    num_slices = int(data[index])
    index += 1
    port_bw_gbps = float(data[index])  # Port Bandwidth in Gbps
    index += 1
    port_bw = port_bw_gbps  # Port Bandwidth for calculations

    slices = []        # List to hold information about each slice
    all_packets = []   # List to hold all packets across slices

    # Process each slice's data
    for slice_id in range(num_slices):
        num_slice_packets = int(data[index])  # Number of packets in the current slice
        index += 1
        slice_bw = float(data[index])         # Slice Bandwidth (Gbps) [Ignored in this strategy]
        index += 1
        slice_delay_tolerance = int(data[index])  # Maximum delay tolerance (UBD)
        index += 1

        # Initialize list to store packets for the current slice
        packets = []
        for packet_id in range(num_slice_packets):
            arrival_time_ns = int(data[index])   # Packet arrival time in nanoseconds
            index += 1
            packet_size = int(data[index])       # Packet size in bits
            index += 1
            deadline = arrival_time_ns + slice_delay_tolerance  # Calculate packet deadline

            # Create a dictionary to represent the packet's attributes
            packet = {
                'ts': arrival_time_ns,            # Arrival time
                'pkt_size': packet_size,          # Packet size
                'pkt_id': packet_id,              # Packet ID within the slice
                'slice_id': slice_id,             # Slice ID
                'deadline': deadline              # Deadline for the packet
            }
            packets.append(packet)
            all_packets.append(packet)

        # Sort packets in the slice based on arrival time and store in a deque for efficient popping
        sorted_packets = deque(sorted(packets, key=lambda pkt: pkt['ts']))

        # Create a dictionary to hold slice-specific information
        slice_info = {
            'ubd': slice_delay_tolerance,       # Maximum delay tolerance
            'slice_bw': slice_bw,               # Slice Bandwidth [Ignored]
            'slice_id': slice_id,               # Slice ID
            'packets': sorted_packets,          # Sorted packets in the slice
            'total_bits_sent': 0,               # Total bits sent for the slice
            'first_ts': packets[0]['ts'] if packets else 0,  # Arrival time of the first packet
            'last_te': 0,                       # Last departure time
            'max_delay': 0,                     # Maximum delay encountered
            'scheduled_packets': []             # List to store scheduled packets for the slice
        }
        slices.append(slice_info)

    # Sort all packets by their arrival time to process them in order
    all_packets.sort(key=lambda pkt: pkt['ts'])
    total_packets = len(all_packets)  # Total number of packets across all slices

    scheduled_packets = []  # List to record the scheduling sequence

    # Initialize scheduling variables
    current_time = 0
    port_available_time = 0
    packet_index = 0
    heap = []  # Min-heap to manage packet priorities

    def get_priority(packet, current_time):
        """
        Calculate the priority of a packet based on its slack time.
        
        Priority is determined by:
        1. Least slack time (higher priority)
        2. Shortest transmission time (higher priority)
        3. Earlier arrival time (higher priority)
        4. Lower slice ID (higher priority)
        
        Args:
            packet (dict): The packet dictionary containing its attributes.
            current_time (int): The current scheduling time.
        
        Returns:
            tuple: A tuple representing the packet's priority.
        """
        transmission_time = math.ceil(packet['pkt_size'] / port_bw)
        slack_time = packet['deadline'] - (current_time + transmission_time)
        return (
            slack_time,                          # Primary: Least slack time
            transmission_time,                   # Secondary: Shortest transmission time
            packet['ts'],                        # Tertiary: Earlier arrival time
            packet['slice_id']                   # Quaternary: Lower slice ID
        )

    # Main scheduling loop
    while packet_index < total_packets or heap:
        # Add all packets that have arrived up to the current_time to the heap
        while (packet_index < total_packets and
               all_packets[packet_index]['ts'] <= current_time):
            packet = all_packets[packet_index]
            priority = get_priority(packet, current_time)
            heapq.heappush(heap, (priority, packet))
            packet_index += 1

        if not heap:
            # No packets to schedule, advance current_time to the next packet's arrival or port availability
            if packet_index < total_packets:
                next_packet_ts = all_packets[packet_index]['ts']
                current_time = max(current_time, next_packet_ts, port_available_time)
            else:
                break
            continue

        # Pop the packet with the highest priority (least slack time) from the heap
        _, packet = heapq.heappop(heap)
        slice_id = packet['slice_id']
        slice_info = slices[slice_id]

        # Ensure packets are scheduled in the order they arrived within the same slice
        if slice_info['packets'] and slice_info['packets'][0]['pkt_id'] != packet['pkt_id']:
            # Reinsert the packet into the heap and fetch the next available packet
            priority = get_priority(packet, current_time)
            heapq.heappush(heap, (priority, packet))
            if slice_info['packets']:
                next_packet = slice_info['packets'][0]
                current_time = max(current_time, next_packet['ts'], port_available_time)
            continue

        # Remove the next packet from the slice's queue
        if slice_info['packets']:
            scheduled_packet = slice_info['packets'].popleft()
        else:
            scheduled_packet = None

        # Determine the earliest possible departure time for the packet
        te_candidate = max(
            current_time,
            packet['ts'],
            slice_info['last_te'],
            port_available_time
        )
        te = math.ceil(te_candidate)

        # Calculate the transmission time based on packet size and port bandwidth
        transmission_time = math.ceil(packet['pkt_size'] / port_bw)

        # Update the port's next available time
        port_available_time = te + transmission_time

        # Update the slice's last departure time
        slice_info['last_te'] = te

        # Update slice metrics
        slice_info['total_bits_sent'] += packet['pkt_size']
        delay = te - packet['ts']
        if delay > slice_info['max_delay']:
            slice_info['max_delay'] = delay

        # Record the scheduled packet's departure time, slice ID, packet ID, and packet size
        scheduled_packets.append((te, slice_id, packet['pkt_id'], packet['pkt_size'], packet['ts']))

        # Also record in slice's scheduled_packets for per-slice checks
        slice_info['scheduled_packets'].append((te, packet['pkt_id'], packet['pkt_size'], packet['ts']))

        # Advance the current_time to the packet's departure time
        current_time = te

        # Early termination if runtime exceeds 2 minutes
        if time.time() - start_time > 120:
            raise Exception("Constraint violated: Runtime exceeds 2 minutes.")

    # Constraint Checks

    # Constraint 1: Port bandwidth constraint (PortBW)
    # Ensure that the time between the departure times of consecutive packets
    # is enough to transmit the previous packet at PortBW.
    for i in range(len(scheduled_packets) - 1):
        te_prev, _, _, pkt_size_prev, _ = scheduled_packets[i]
        te_next, _, _, _, _ = scheduled_packets[i + 1]
        # Transmission time of the previous packet
        transmission_time_prev = math.ceil(pkt_size_prev / port_bw)
        # The earliest the next packet can be scheduled
        min_te_next = te_prev + transmission_time_prev
        if te_next < min_te_next:
            raise Exception(f"Constraint 1 violated: Port bandwidth constraint not met between packet indices {i} and {i+1}.")

    # Constraint 3: Packet scheduling sequence constraint
    # Packets in the same slice must leave in the order they arrived
    # and the packet leaving time must be longer than the packet arrival time.
    for slice_info in slices:
        scheduled_packets_slice = slice_info['scheduled_packets']
        for j in range(len(scheduled_packets_slice)):
            te_current, pkt_id_current, _, ts_current = scheduled_packets_slice[j]
            # Check that te_current >= ts_current
            if te_current < ts_current:
                raise Exception(f"Constraint 3 violated: Packet departure time is before arrival time for slice {slice_info['slice_id']}, packet {pkt_id_current}.")
            if j > 0:
                te_prev, pkt_id_prev, _, _ = scheduled_packets_slice[j - 1]
                # Check that te_current >= te_prev
                if te_current < te_prev:
                    raise Exception(f"Constraint 3 violated: Packet departure times are not non-decreasing within slice {slice_info['slice_id']} between packets {pkt_id_prev} and {pkt_id_current}.")

    # Constraint 4: Ensure all timestamps (te and ts) are integers
    # This check ensures that all departure times (te) and arrival times (ts) are integers.
    if not all(isinstance(te, int) for te, _, _, _, _ in scheduled_packets):
        raise Exception("Constraint 4 violated: All packet departure times (te) must be integers.")

    if not all(isinstance(packet['ts'], int) for packet in all_packets):
        raise Exception("Constraint 4 violated: All packet arrival times (ts) must be integers.")

    # Constraint: The Output Slice ID and Packet ID must start from 0
    # This check ensures that the minimum Slice ID and Packet ID in the output are 0.
    min_slice_id = min(slice_id for _, slice_id, _, _, _ in scheduled_packets)
    min_pkt_id = min(pkt_id for _, _, pkt_id, _, _ in scheduled_packets)
    if min_slice_id != 0 or min_pkt_id != 0:
        raise Exception("Constraint violated: Output Slice ID and Packet ID must start from 0.")

    # Constraint: Number of scheduled packets must equal number of input packets
    # This check ensures that all input packets have been scheduled.
    if len(scheduled_packets) != total_packets:
        raise Exception("Constraint violated: Number of scheduled packets does not equal number of input packets.")

    # Calculate the maximum delay across all slices
    max_delay = 0
    for slice_info in slices:
        if slice_info['max_delay'] > max_delay:
            max_delay = slice_info['max_delay']

    # Calculate the score based on the scheduling performance
    # Since we're ignoring FI, the score is solely based on max_delay
    # Assuming lower max_delay results in a higher score
    # The exact scoring mechanism should be defined; here's a placeholder
    score = 10000 / max_delay if max_delay > 0 else 10000

    # End time for runtime constraint checking
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Constraint: Runtime has to be less than 2 minutes
    # This check ensures that the total execution time does not exceed 2 minutes (120 seconds).
    if elapsed_time > 120:
        raise Exception("Constraint violated: Runtime exceeds 2 minutes.")

    # Output generation
    # K must equal the number of scheduled packets
    K = len(scheduled_packets)
    if K == total_packets:
        print(K)
    else:
        raise Exception("Constraint violated: Number of scheduled packets does not match input packets.")

    # Prepare the output scheduling sequence
    output = []
    for te, slice_id, pkt_id, _, _ in scheduled_packets:
        output.append(f"{te} {slice_id} {pkt_id}")
    print(' '.join(output))

if __name__ == "__main__":
    # Start the main function in a separate thread to avoid potential recursion limits
    threading.Thread(target=main).start()