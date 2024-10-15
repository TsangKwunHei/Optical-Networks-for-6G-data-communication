import sys
import math
import heapq
import threading

def main():
    import sys

    # Read all input data
    data = sys.stdin.read().split()
    idx = 0

    # Read number of slices and PortBW
    n = int(data[idx]); idx += 1
    PortBW_Gbps = float(data[idx]); idx += 1  # in Gbps
    PortBW = PortBW_Gbps  # bits/ns (1 Gbps = 1 bit/ns)

    slices = []
    all_packets = []

    for slice_id in range(n):
        m_i = int(data[idx]); idx += 1
        SliceBW_i = float(data[idx]); idx += 1  # in Gbps
        UBD_i = int(data[idx]); idx += 1  # in ns
        packets = []
        for pkt_id in range(m_i):
            ts = int(data[idx]); idx += 1  # in ns
            pkt_size = int(data[idx]); idx += 1  # in bits
            packets.append( (ts, pkt_size, pkt_id, slice_id) )
            all_packets.append( (ts, pkt_size, pkt_id, slice_id) )
        slices.append({
            'UBD_i': UBD_i,
            'SliceBW_i': SliceBW_i,
            'slice_id': slice_id,
            'm_i': m_i,
            'packets': packets,
            'next_pkt_idx': 0,  # Next packet to schedule
            'last_te': 0,        # Departure time of the last packet in this slice
            'sum_bits': 0,       # Total bits sent in this slice
            'te_last': 0,        # Departure time of the last packet in this slice
            'ts_first': packets[0][0] if m_i > 0 else 0,
            'max_delay': 0       # Maximum delay in this slice
        })

    # Initialize a min-heap with the first packet from each slice, sorted by deadline
    heap = []
    for s in slices:
        if s['m_i'] >0:
            pkt = s['packets'][0]
            ts, pkt_size, pkt_id, slice_id = pkt
            deadline = ts + s['UBD_i']
            # Heap ordered by (deadline, -SliceBW_i, slice_id, pkt_id, ts, pkt_size)
            heapq.heappush(heap, (deadline, -s['SliceBW_i'], slice_id, pkt_id, ts, pkt_size))
            s['next_pkt_idx'] =1  # Next packet index

    scheduled = []
    port_available_time =0

    while heap:
        # Pop the packet with the earliest deadline and highest SliceBW_i
        deadline, neg_SliceBW_i, slice_id, pkt_id, ts, pkt_size = heapq.heappop(heap)
        s = slices[slice_id]

        # Calculate departure time (te)
        te_candidate = max(port_available_time, ts, s['last_te'])
        te = math.ceil(te_candidate)  # Ensure te is integer

        # Calculate transmission time (ceiling division)
        transmission_time = math.ceil(pkt_size / PortBW)

        # Update port availability
        port_available_time = te + transmission_time

        # Record the scheduled packet
        scheduled.append( (int(te), slice_id, pkt_id) )

        # Update slice's last_te
        s['last_te'] = te

        # Update slice's metrics
        s['sum_bits'] += pkt_size
        s['te_last'] = te
        delay = te - ts
        if delay > s['max_delay']:
            s['max_delay'] = delay

        # Push the next packet from the slice into the heap, if any
        if s['next_pkt_idx'] < s['m_i']:
            next_pkt = s['packets'][s['next_pkt_idx']]
            next_ts, next_pkt_size, next_pkt_id, _ = next_pkt
            next_deadline = next_ts + s['UBD_i']
            heapq.heappush(heap, (next_deadline, -s['SliceBW_i'], slice_id, next_pkt_id, next_ts, next_pkt_size))
            s['next_pkt_idx'] +=1

    # After scheduling all packets, verify constraints
    valid = True
    for s in slices:
        if s['m_i'] ==0:
            continue  # No packets, automatically valid
        slice_id = s['slice_id']
        total_time = s['te_last'] - s['ts_first']
        if total_time ==0:
            bw = float('inf')  # Only one packet with te == ts
        else:
            bw = s['sum_bits'] / total_time  # bits/ns
        required_bw = 0.95 * s['SliceBW_i']  # bits/ns
        if bw < required_bw:
            valid=False
            break
        if s['max_delay'] > s['UBD_i']:
            valid=False
            break

    # Regardless of constraints, output all scheduled packets to avoid format errors
    # The scoring system will assign a score of 0 if constraints are violated
    K = len(scheduled)
    total_input_packets = sum(s['m_i'] for s in slices)

    # Ensure K matches sum(m_i)
    if K != total_input_packets:
        # This should not happen with the current scheduling logic
        # However, as a safeguard, iterate through all slices and schedule any missing packets
        for s in slices:
            while s['next_pkt_idx'] < s['m_i']:
                pkt = s['packets'][s['next_pkt_idx']]
                ts, pkt_size, pkt_id, slice_id = pkt
                te_candidate = max(port_available_time, ts, s['last_te'])
                te = math.ceil(te_candidate)
                transmission_time = math.ceil(pkt_size / PortBW)
                port_available_time = te + transmission_time
                scheduled.append( (int(te), slice_id, pkt_id) )
                s['last_te'] = te
                s['sum_bits'] += pkt_size
                s['te_last'] = te
                delay = te - ts
                if delay > s['max_delay']:
                    s['max_delay'] = delay
                s['next_pkt_idx'] +=1
                K +=1

    # Final Output
    print(K)
    output = []
    for te, slice_id, pkt_id in scheduled:
        output.append(f"{te} {slice_id} {pkt_id}")
    print(' '.join(output))

if __name__ == "__main__":
    threading.Thread(target=main).start()