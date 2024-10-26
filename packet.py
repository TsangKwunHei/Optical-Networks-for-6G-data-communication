from dataclasses import dataclass
import sys
import math
import heapq
from collections import deque
from itertools import count  # Import count for unique sequence numbers


@dataclass
class Packet:
    ts: int
    pkt_size: int
    pkt_id: int
    slice_id: int
    deadline: int


@dataclass
class Slice:
    ubd: int
    slice_bw: float
    slice_id: int
    packets: deque[Packet]
    total_bits_sent: int
    first_ts: int
    last_te: int
    max_delay: int
    expected_bits: int = 0  # Track expected bandwidth usage
    waiting_time: int = 0  # Track cumulative waiting time


def get_environment():
    # Read all input data from standard input and split into tokens
    data = sys.stdin.read().split()
    index = 0  # Current position in the input data

    # Parse the number of slices and Port Bandwidth (Gbps)
    num_slices = int(data[index])
    index += 1
    port_bw_gbps = float(data[index])  # Port Bandwidth in Gbps
    index += 1
    port_bw = port_bw_gbps  # Port Bandwidth for calculations

    slices: list[Slice] = []  # List to hold all slices
    all_packets: list[Packet] = []  # List to hold all packets across all slices

    # Process each slice's data
    for slice_id in range(num_slices):
        num_slice_packets = int(data[index])  # Number of packets in the current slice
        index += 1
        slice_bw = float(data[index])  # Slice Bandwidth (Gbps)
        index += 1
        slice_delay_tolerance = int(data[index])  # Maximum delay tolerance (UBD)
        index += 1

        # Initialize list to store packets for the current slice
        packets: list[Packet] = []
        for packet_id in range(num_slice_packets):
            arrival_time_ns = int(data[index])  # Packet arrival time in nanoseconds
            index += 1
            packet_size = int(data[index])  # Packet size in bits
            index += 1
            deadline = (
                arrival_time_ns + slice_delay_tolerance
            )  # Calculate packet deadline

            # Create a dictionary to represent the packet's attributes
            packet = Packet(
                ts=arrival_time_ns,
                pkt_size=packet_size,
                pkt_id=packet_id,
                slice_id=slice_id,
                deadline=deadline,
            )
            packets.append(packet)
            all_packets.append(packet)

        # Sort packets in the slice based on arrival time and store in a deque for efficient popping
        sorted_packets = deque(sorted(packets, key=lambda pkt: pkt.ts))

        # Create a dictionary to hold slice-specific information

        slice_info = Slice(
            ubd=slice_delay_tolerance,
            slice_bw=slice_bw,
            slice_id=slice_id,
            packets=sorted_packets,
            total_bits_sent=0,
            first_ts=packets[0].ts if packets else 0,
            last_te=0,
            max_delay=0,
        )
        slices.append(slice_info)

    # Sort all packets by their arrival time to process them in order
    all_packets.sort(key=lambda pkt: pkt.ts)
    return port_bw, slices, all_packets


def get_slice_utilization(slice_info: Slice, current_time: int) -> float:
    """Calculate slice utilization ratio"""
    elapsed_time = max(1, current_time - slice_info.first_ts)
    expected_bits = (
        slice_info.slice_bw * elapsed_time
    )  # Expected bits based on bandwidth
    actual_bits = slice_info.total_bits_sent
    return actual_bits / expected_bits if expected_bits > 0 else 1.0


def calculate_urgency(packet: Packet, current_time: int) -> float:
    """Calculate packet urgency based on deadline and size"""
    remaining_time = max(1, packet.deadline - current_time)
    return packet.pkt_size / remaining_time


def get_aging_factor(
    packet: Packet, current_time: int, base_aging: float = 1.2
) -> float:
    """Calculate aging factor that increases priority for waiting packets"""
    waiting_time = current_time - packet.ts
    return base_aging ** (waiting_time // 1000)


def get_priority(packet: Packet, slice_info: Slice, current_time: int, cnt: int):
    """Enhanced priority calculation incorporating multiple factors"""
    # Calculate basic urgency
    urgency = calculate_urgency(packet, current_time)

    # Get slice utilization (lower utilization = higher priority)
    utilization = get_slice_utilization(slice_info, current_time)

    # Calculate normalized deadline factor (earlier deadline = higher priority)
    deadline_factor = 1.0 / max(1, packet.deadline - current_time)

    # Get aging factor
    aging = get_aging_factor(packet, current_time)

    # Calculate final priority score (lower value = higher priority)
    priority_score = (
        deadline_factor,
        -urgency * aging,  # Urgency adjusted by aging
        -slice_info.slice_bw
        * (1 - utilization),  # Bandwidth weight adjusted by utilization
        cnt,  # Maintain stable sorting
    )

    return priority_score


def main():
    port_bw, slices, all_packets = get_environment()
    total_packets = len(all_packets)  # Total number of packets
    scheduled_packets: list[
        tuple[int, int, int]
    ] = []  # List to store scheduled packets

    # Initialize scheduling variables
    current_time = 0
    port_available_time = 0
    packet_index = 0
    heap: list[tuple[tuple[float, float, float, int], Packet]] = []  # Priority heap
    unique_counter = count()  # Initialize the unique counter

    # Main scheduling loop
    while packet_index < total_packets or heap:
        for slice_info in slices:
            if slice_info.packets:
                slice_info.waiting_time += len(slice_info.packets)
        # Add all packets that have arrived up to the current_time to the heap
        while (
            packet_index < total_packets
            and all_packets[packet_index].ts <= current_time
        ):
            packet = all_packets[packet_index]
            cnt = next(unique_counter)  # Get the next unique counter
            heapq.heappush(
                heap,
                (
                    get_priority(packet, slices[packet.slice_id], current_time, cnt),
                    packet,
                ),
            )
            packet_index += 1

        if not heap:
            # No packets to schedule, advance current_time to the next packet's arrival or port availability
            if packet_index < total_packets:
                next_packet_ts = all_packets[packet_index].ts
                current_time = max(current_time, next_packet_ts, port_available_time)
            else:
                break
            continue

        # Pop the packet with the highest priority from the heap
        _, packet = heapq.heappop(heap)
        slice_id = packet.slice_id
        slice_info = slices[slice_id]

        # Ensure packets are scheduled in the order they arrived within the same slice
        if slice_info.packets and slice_info.packets[0].pkt_id != packet.pkt_id:
            heapq.heappush(
                heap,
                (
                    get_priority(
                        packet,
                        slices[packet.slice_id],
                        current_time,
                        next(unique_counter),
                    ),
                    packet,
                ),
            )
            if slice_info.packets:
                next_packet = slice_info.packets[0]
                current_time = max(current_time, next_packet.ts, port_available_time)
            continue

        # Remove the next packet from the slice's queue
        if slice_info.packets:
            next_packet = slice_info.packets.popleft()
        else:
            next_packet = None

        # Determine the earliest possible departure time for the packet
        te_candidate = max(
            current_time, packet.ts, slice_info.last_te, port_available_time
        )
        te = math.ceil(te_candidate)

        """
        In this code, 
        now it basically every time it's just choose the, 
        port_available_time as te, 
        (except that for 1st packet it process (which port_available_time is not yet available) choose the current's time)
        """
        # Calculate the transmission time based on packet size and port bandwidth
        transmission_time = math.ceil(packet.pkt_size / port_bw)

        # Update the port's next available time
        port_available_time = te + transmission_time

        # Update the slice's last departure time
        slice_info.last_te = te

        # Update slice metrics
        slice_info.total_bits_sent += packet.pkt_size
        delay = te - packet.ts
        if delay > slice_info.max_delay:
            slice_info.max_delay = delay

        # Record the scheduled packet's departure time, slice ID, and packet ID
        scheduled_packets.append((te, slice_id, packet.pkt_id))

        # Advance the current_time to the packet's departure time
        current_time = te

    # Calculate the score based on the scheduling performance
    # fi_sum = 0
    # max_delay = 0
    # for slice_info in slices:
    #     # Determine if the slice meets its delay tolerance
    #     fi = 1 if slice_info.max_delay <= slice_info.ubd else 0
    #     fi_sum += fi / num_slices
    #     if slice_info.max_delay > max_delay:
    #         max_delay = slice_info.max_delay

    # Section I made for debug (Useless to actual Output)
    # if max_delay > 0:
    #     score = fi_sum + (10000 / max_delay)
    # else:
    #     score = fi_sum + 10000

    # Output generation
    # K must equal the number of scheduled packets
    K = len(scheduled_packets)
    print(K)

    # Prepare the output scheduling sequence
    output: list[str] = []
    for te, slice_id, pkt_id in scheduled_packets:
        output.append(f"{te} {slice_id} {pkt_id}")
    print(" ".join(output))


if __name__ == "__main__":
    # Start a thread with timeout of 20 seconds. Get stacktrace of where the code is stuck
    import threading

    def timeout():
        import traceback

        print("Timeout")
        traceback.print_stack()
        sys.exit(1)

    timer = threading.Timer(20, timeout)

    timer.start()
    main()

    timer.cancel()
    sys.exit(0)
