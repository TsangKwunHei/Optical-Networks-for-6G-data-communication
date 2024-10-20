import sys
import threading
import math
import heapq
from collections import deque


def main():
    """
    Main function to execute the soft slice scheduling algorithm.
    
    This function reads input data, processes packet scheduling across multiple slices,
    and outputs the scheduling sequence along with relevant metrics.
    
    The scheduling aims to maximize the number of slices meeting their bandwidth and delay constraints
    while minimizing the maximum packet delay.
    """
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
        slice_bw = float(data[index])         # Slice Bandwidth (Gbps)
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
            'slice_bw': slice_bw,               # Slice Bandwidth
            'slice_id': slice_id,               # Slice ID
            'packets': sorted_packets,          # Sorted packets in the slice
            'total_bits_sent': 0,               # Total bits sent for the slice
            'first_ts': packets[0]['ts'] if packets else 0,  # Arrival time of the first packet
            'last_te': 0,                       # Last departure time
            'max_delay': 0                      # Maximum delay encountered
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

    def get_priority(packet):
        """
        Calculate the priority of a packet based on its deadline, slice bandwidth, and slice ID.
        
        Priority is determined by:
        1. Earliest deadline (higher priority)
        2. Higher slice bandwidth (negated for max-heap behavior)
        3. Lower slice ID (earlier slices have higher priority)
        
        Args:
            packet (dict): The packet dictionary containing its attributes.
        
        Returns:
            tuple: A tuple representing the packet's priority.
        """
        return (
            packet['deadline'],
            -slices[packet['slice_id']]['slice_bw'],
            slices[packet['slice_id']]['slice_id']
        )

    # Main scheduling loop
    while packet_index < total_packets or heap:
        # Add all packets that have arrived up to the current_time to the heap
        while (packet_index < total_packets and
               all_packets[packet_index]['ts'] <= current_time):
            packet = all_packets[packet_index]
            heapq.heappush(heap, (get_priority(packet), packet))
            packet_index += 1

        if not heap:
            # No packets to schedule, advance current_time to the next packet's arrival or port availability
            if packet_index < total_packets:
                next_packet_ts = all_packets[packet_index]['ts']
                current_time = max(current_time, next_packet_ts, port_available_time)
            else:
                break
            continue

        # Pop the packet with the highest priority from the heap
        _, packet = heapq.heappop(heap)
        slice_id = packet['slice_id']
        slice_info = slices[slice_id]

        # Ensure packets are scheduled in the order they arrived within the same slice
        if slice_info['packets'] and slice_info['packets'][0]['pkt_id'] != packet['pkt_id']:
            heapq.heappush(heap, (get_priority(packet), packet))
            if slice_info['packets']:
                next_packet = slice_info['packets'][0]
                current_time = max(current_time, next_packet['ts'], port_available_time)
            continue

        # Remove the next packet from the slice's queue
        if slice_info['packets']:
            next_packet = slice_info['packets'].popleft()
        else:
            next_packet = None

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

        # Record the scheduled packet's departure time, slice ID, and packet ID
        scheduled_packets.append((te, slice_id, packet['pkt_id']))

        # Advance the current_time to the packet's departure time
        current_time = te

    # Calculate the score based on the scheduling performance
    fi_sum = 0
    max_delay = 0
    for slice_info in slices:
        # Determine if the slice meets its delay tolerance
        fi = 1 if slice_info['max_delay'] <= slice_info['ubd'] else 0
        fi_sum += fi / num_slices
        if slice_info['max_delay'] > max_delay:
            max_delay = slice_info['max_delay']

    # Avoid division by zero by handling cases where max_delay is 0
    if max_delay > 0:
        score = fi_sum + (10000 / max_delay)
    else:
        score = fi_sum + 10000

    # Output generation
    # K must equal the number of scheduled packets
    K = len(scheduled_packets)
    print(K)

    # Prepare the output scheduling sequence
    output = []
    for te, slice_id, pkt_id in scheduled_packets:
        output.append(f"{te} {slice_id} {pkt_id}")
    print(' '.join(output))


if __name__ == "__main__":
    # Start the main function in a separate thread to avoid potential recursion limits
    threading.Thread(target=main).start()
