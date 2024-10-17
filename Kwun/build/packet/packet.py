import sys
import threading
import heapq

def main():
    import math

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    # Read the first line: number of slices and PortBW
    n, PortBW = sys.stdin.readline().split()
    n = int(n)
    PortBW = float(PortBW) * 1e9  # Convert Gbps to bps

    slices = []
    for slice_id in range(n):
        m_i, SliceBW_i, UBD_i = sys.stdin.readline().split()
        m_i = int(m_i)
        SliceBW_i = float(SliceBW_i) * 1e9  # Convert Gbps to bps
        UBD_i = int(UBD_i)  # in ns

        # Read the packet sequence information
        tokens = sys.stdin.readline().split()
        ts_list = [int(tokens[i]) for i in range(0, len(tokens), 2)]
        PktSize_list = [int(tokens[i]) for i in range(1, len(tokens), 2)]

        # Precompute accumulated PktSize for bandwidth constraints
        accumulated_PktSize = []
        total_PktSize = 0
        for size in PktSize_list:
            total_PktSize += size
            accumulated_PktSize.append(total_PktSize)

        slice_info = {
            'slice_id': slice_id,
            'm_i': m_i,
            'SliceBW_i': SliceBW_i,
            'UBD_i': UBD_i,
            'ts_list': ts_list,
            'PktSize_list': PktSize_list,
            'accumulated_PktSize': accumulated_PktSize,
            'pkt_index': 0,  # Index of the next packet to schedule
            'te_list': [0] * m_i,
            'total_bits_transmitted': 0,
            'Delay_i': 0,
            'ts_first': ts_list[0],
            'te_last': 0
        }
        slices.append(slice_info)

    # Initialize time and previous transmission info
    t = 0
    te_prev = 0
    transmission_time_prev = 0

    # Initialize min-heap of arrived packets
    heap = []
    arrived_slices = set()
    pending_slices = set(range(n))

    # Initialize packet arrival events
    packet_events = []
    for s in slices:
        ts = s['ts_list'][0]
        packet_events.append((ts, s['slice_id']))

    packet_events.sort()

    # While there are packets to schedule
    total_packets = sum(s['m_i'] for s in slices)
    scheduled_packets = 0
    output_sequence = []

    while scheduled_packets < total_packets:
        # If no arrived packets, advance time to next arrival
        if not heap:
            if packet_events:
                next_arrival_time, slice_id = packet_events.pop(0)
                t = max(t, next_arrival_time)
                # Add the packet to the heap
                s = slices[slice_id]
                pkt_idx = s['pkt_index']
                ts = s['ts_list'][pkt_idx]
                PktSize = s['PktSize_list'][pkt_idx]
                # Compute deadlines
                Deadline1 = ts + s['UBD_i']
                Accumulated_PktSize = s['accumulated_PktSize'][pkt_idx]
                Deadline2 = s['ts_first'] + Accumulated_PktSize / (0.95 * s['SliceBW_i']) * 1e9  # Convert to ns
                Effective_Deadline = min(Deadline1, Deadline2)
                heapq.heappush(heap, (Effective_Deadline, slice_id))
                arrived_slices.add(slice_id)
            else:
                break  # No more packets to schedule
        else:
            # Schedule the packet with the earliest Effective Deadline
            Effective_Deadline, slice_id = heapq.heappop(heap)
            s = slices[slice_id]
            pkt_idx = s['pkt_index']
            ts = s['ts_list'][pkt_idx]
            PktSize = s['PktSize_list'][pkt_idx]
            transmission_time = PktSize / PortBW * 1e9  # in ns
            te_candidate = max(t, ts, te_prev + transmission_time_prev)
            te_i_j = int(math.ceil(te_candidate))
            s['te_list'][pkt_idx] = te_i_j
            # Update t and previous transmission info
            t = te_i_j
            te_prev = te_i_j
            transmission_time_prev = transmission_time
            # Update total bits transmitted for the slice
            s['total_bits_transmitted'] += PktSize
            # Update Delay_i for the slice
            packet_delay = te_i_j - ts
            s['Delay_i'] = max(s['Delay_i'], packet_delay)
            # Update te_last for the slice
            s['te_last'] = te_i_j
            # Prepare output
            output_sequence.append((te_i_j, slice_id, pkt_idx))
            scheduled_packets += 1
            # Move to next packet in the slice
            s['pkt_index'] += 1
            if s['pkt_index'] < s['m_i']:
                # If next packet has arrived, add to heap
                next_pkt_idx = s['pkt_index']
                ts_next = s['ts_list'][next_pkt_idx]
                if ts_next <= t:
                    PktSize_next = s['PktSize_list'][next_pkt_idx]
                    # Compute deadlines
                    Deadline1 = ts_next + s['UBD_i']
                    Accumulated_PktSize = s['accumulated_PktSize'][next_pkt_idx]
                    Deadline2 = s['ts_first'] + Accumulated_PktSize / (0.95 * s['SliceBW_i']) * 1e9  # ns
                    Effective_Deadline = min(Deadline1, Deadline2)
                    heapq.heappush(heap, (Effective_Deadline, slice_id))
                else:
                    # Schedule an event for the packet arrival
                    heapq.heappush(packet_events, (ts_next, slice_id))
                    packet_events.sort()
            else:
                arrived_slices.remove(slice_id)

    # After scheduling, check constraints
    constraints_met = True
    for s in slices:
        # Check per-slice bandwidth constraint
        duration = s['te_last'] - s['ts_first']
        achieved_bandwidth = s['total_bits_transmitted'] / duration if duration > 0 else float('inf')
        required_bandwidth = 0.95 * s['SliceBW_i'] / 1e9  # in bits/ns
        if achieved_bandwidth < required_bandwidth:
            constraints_met = False
            break
        # Check packet ordering and time constraints within the slice
        for idx in range(1, s['m_i']):
            if s['te_list'][idx] < s['te_list'][idx - 1]:
                constraints_met = False
                break
            if s['te_list'][idx] < s['ts_list'][idx]:
                constraints_met = False
                break
        if not constraints_met:
            break

    if not constraints_met:
        print(0)
        return

    # Output
    print(scheduled_packets)
    output_line = []
    for te_i_j, slice_id, pkt_idx in output_sequence:
        output_line.extend([str(te_i_j), str(slice_id), str(pkt_idx)])
    print(' '.join(output_line))

if __name__ == "__main__":
    threading.Thread(target=main).start()