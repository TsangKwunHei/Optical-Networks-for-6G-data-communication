import sys
import threading
import math
import heapq
from collections import deque

def main():
    import sys

    # Read all input data
    data = sys.stdin.read().split()
    idx = 0

    # Number of slices and PortBW
    n = int(data[idx]); idx += 1
    PortBW_Gbps = float(data[idx]); idx += 1  # in Gbps
    PortBW = PortBW_Gbps  # bits/ns (since 1 Gbps = 1 bit/ns)

    slices = []
    all_packets = []
    for slice_id in range(n):
        m_i = int(data[idx]); idx += 1
        SliceBW_i = float(data[idx]); idx += 1  # in Gbps
        UBD_i = int(data[idx]); idx += 1  # in ns
        packets = []
        for pkt_id in range(m_i):
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
        slice_info = {
            'UBD_i': UBD_i,
            'SliceBW_i': SliceBW_i,
            'slice_id': slice_id,
            'packets': deque(sorted(packets, key=lambda x: x['ts'])),  # Ensure packets are ordered
            'total_bits_sent': 0,
            'first_ts': packets[0]['ts'] if packets else 0,
            'last_te': 0,
            'max_delay': 0
        }
        slices.append(slice_info)

    # Sort all packets by arrival time
    all_packets.sort(key=lambda pkt: pkt['ts'])
    total_packets = len(all_packets)

    # Scheduled packets list
    scheduled_packets = []

    # Initialize variables
    current_time = 0
    port_available_time = 0
    packet_idx = 0
    heap = []

    # Priority function: (earliest deadline, highest SliceBW, earliest slice_id)
    def get_priority(pkt):
        return (pkt['deadline'], -slices[pkt['slice_id']]['SliceBW_i'], slices[pkt['slice_id']]['slice_id'])

    # Main scheduling loop
    while packet_idx < total_packets or heap:
        # Add all packets that have arrived up to current_time to the heap
        while packet_idx < total_packets and all_packets[packet_idx]['ts'] <= current_time:
            pkt = all_packets[packet_idx]
            heapq.heappush(heap, (get_priority(pkt), pkt))
            packet_idx += 1

        if not heap:
            # No packets to schedule, advance time
            if packet_idx < total_packets:
                current_time = max(current_time, all_packets[packet_idx]['ts'], port_available_time)
            else:
                break
            continue

        # Pop the packet with the highest priority
        _, pkt = heapq.heappop(heap)
        slice_id = pkt['slice_id']
        s = slices[slice_id]

        # Ensure packets are scheduled in order within the slice
        if s['packets'] and s['packets'][0]['pkt_id'] != pkt['pkt_id']:
            # Not the next packet in slice, skip and re-add to heap
            heapq.heappush(heap, ((get_priority(pkt)), pkt))
            # Advance current_time to the earliest possible next packet
            if s['packets']:
                next_pkt = s['packets'][0]
                current_time = max(current_time, next_pkt['ts'], port_available_time)
            continue

        # Pop the next packet from the slice's queue
        if s['packets']:
            next_packet = s['packets'].popleft()
        else:
            next_packet = None

        # Determine earliest possible departure time
        te_candidate = max(current_time, pkt['ts'], s['last_te'], port_available_time)
        te = math.ceil(te_candidate)

        # Calculate transmission time
        transmission_time = math.ceil(pkt['pkt_size'] / PortBW)

        # Update port availability
        port_available_time = te + transmission_time

        # Update slice's last_te
        s['last_te'] = te

        # Update slice's metrics
        s['total_bits_sent'] += pkt['pkt_size']
        delay = te - pkt['ts']
        if delay > s['max_delay']:
            s['max_delay'] = delay

        # Record the scheduled packet
        scheduled_packets.append((te, slice_id, pkt['pkt_id']))

        # Advance current time
        current_time = te

    # After scheduling, verify and adjust to meet slice bandwidth constraints
    # This section can be expanded to adjust scheduling if necessary

    # Output generation
    K = len(scheduled_packets)
    print(K)
    output = []
    for te, slice_id, pkt_id in scheduled_packets:
        output.append(f"{te} {slice_id} {pkt_id}")
    print(' '.join(output))

if __name__ == "__main__":
    threading.Thread(target=main).start()
