import sys
import heapq

def main():
    import threading
    threading.Thread(target=run).start()

def run():
    import sys
    import math

    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.read().split()
    idx = 0

    n = int(input_data[idx])
    idx += 1
    PortBW = float(input_data[idx]) * 1e9  # Convert Gbps to bps
    idx += 1

    slices = []
    sum_m_i = 0  # Total number of input packets
    for slice_id in range(n):
        m_i = int(input_data[idx])
        idx += 1
        sum_m_i += m_i
        SliceBW_i = float(input_data[idx]) * 1e9  # Convert Gbps to bps
        idx += 1
        UBD_i = int(input_data[idx])  # in ns
        idx += 1

        ts_list = []
        PktSize_list = []
        for _ in range(m_i):
            ts_i_j = int(input_data[idx])  # in ns
            idx += 1
            PktSize_i_j = int(input_data[idx])  # in bits
            idx += 1
            ts_list.append(ts_i_j)
            PktSize_list.append(PktSize_i_j)

        slice_info = {
            'SliceID': slice_id,
            'm_i': m_i,
            'SliceBW_i': SliceBW_i,
            'UBD_i': UBD_i,
            'ts_list': ts_list,
            'PktSize_list': PktSize_list,
            'te_list': [0] * m_i,
            'Delay_i': 0,
            'TotalSize': sum(PktSize_list),
            'S_i_sent': 0,
            'f_i': 1,  # Assume f_i = 1 initially
            'next_pkt_idx': 0,  # Next packet index to schedule
        }
        slices.append(slice_info)

    # Create a list of packet arrivals
    packet_arrivals = []
    for slice_info in slices:
        SliceID = slice_info['SliceID']
        m_i = slice_info['m_i']
        for PacketID in range(m_i):
            ts = slice_info['ts_list'][PacketID]
            packet = {
                'ts': ts,
                'PktSize': slice_info['PktSize_list'][PacketID],
                'SliceID': SliceID,
                'PacketID': PacketID,
                'Deadline': ts + slices[SliceID]['UBD_i']
            }
            packet_arrivals.append(packet)

    # Sort packet arrivals by arrival time
    packet_arrivals.sort(key=lambda x: x['ts'])
    total_packets = len(packet_arrivals)
    arrival_idx = 0  # Index for packet_arrivals

    ready_queue = []  # Min-heap based on Deadline

    te_prev = 0  # Previous packet's departure time
    PktSize_prev = 0  # Previous packet's size
    transmission_time_prev = 0  # Transmission time of previous packet

    last_te_i = {}  # Last departure time for each slice

    t = 0  # Current time

    output_schedule = []

    packets_scheduled = 0

    while packets_scheduled < total_packets or ready_queue:
        # Advance t to next packet's arrival time if ready_queue is empty
        if not ready_queue and arrival_idx < total_packets and packet_arrivals[arrival_idx]['ts'] > t:
            t = packet_arrivals[arrival_idx]['ts']

        # Add all packets that have arrived to the ready_queue
        while arrival_idx < total_packets and packet_arrivals[arrival_idx]['ts'] <= t:
            packet = packet_arrivals[arrival_idx]
            arrival_idx += 1
            # Push to ready_queue based on earliest Deadline
            heapq.heappush(ready_queue, (packet['Deadline'], packet))

        if ready_queue:
            # Select packet with earliest Deadline
            _, packet = heapq.heappop(ready_queue)
            ts_current = packet['ts']
            PktSize_current = packet['PktSize']
            SliceID = packet['SliceID']
            PacketID = packet['PacketID']
            Deadline = packet['Deadline']

            # Calculate te_candidate
            transmission_time_prev = math.ceil(PktSize_prev * 1e9 / PortBW) if PktSize_prev > 0 else 0
            te_candidate = max(
                ts_current,
                te_prev + transmission_time_prev,
                last_te_i.get(SliceID, 0)
            )

            # Ensure te_candidate does not exceed Deadline
            if te_candidate > Deadline:
                # Cannot schedule this packet within its deadline
                # For the purpose of this problem, we'll still schedule it but acknowledge the violation
                # Alternatively, you can handle it as an error or implement a different strategy
                pass  # Proceed to schedule despite deadline violation

            # Assign te_candidate
            slices[SliceID]['te_list'][PacketID] = te_candidate
            # Update Delay_i
            delay = te_candidate - ts_current
            slices[SliceID]['Delay_i'] = max(slices[SliceID]['Delay_i'], delay)
            # Update S_i_sent
            slices[SliceID]['S_i_sent'] += PktSize_current

            # Update previous packet info
            te_prev = te_candidate
            PktSize_prev = PktSize_current
            transmission_time_prev = math.ceil(PktSize_prev * 1e9 / PortBW)

            # Update last_te_i for the current slice
            last_te_i[SliceID] = te_candidate

            output_schedule.append((te_candidate, SliceID, PacketID))
            packets_scheduled += 1

            # Advance time to te_candidate
            t = te_candidate
        else:
            if arrival_idx < total_packets:
                # Advance time to next packet's arrival time
                t = packet_arrivals[arrival_idx]['ts']
            else:
                # No more packets to schedule
                break

    # Compute f_i for each slice
    max_Delay_i = 0
    for slice_info in slices:
        m_i = slice_info['m_i']
        if m_i == 0:
            continue  # Skip slices with no packets
        te_i_m = slice_info['te_list'][-1]
        ts_i_1 = slice_info['ts_list'][0]
        T_i = te_i_m - ts_i_1  # Total time for the slice
        S_i = slice_info['TotalSize']
        BW_i = S_i / T_i if T_i > 0 else float('inf')
        required_BW = 0.95 * slice_info['SliceBW_i']
        Delay_i = slice_info['Delay_i']
        UBD_i = slice_info['UBD_i']
        max_Delay_i = max(max_Delay_i, Delay_i)
        if BW_i >= required_BW and Delay_i <= UBD_i:
            slice_info['f_i'] = 1
        else:
            slice_info['f_i'] = 0

    n_valid_slices = sum(1 for s in slices if s['m_i'] > 0)
    if n_valid_slices > 0:
        sum_f_i = sum(slice_info['f_i'] for slice_info in slices if slice_info['m_i'] > 0)
        Score = (sum_f_i / n_valid_slices) + (10000 / max_Delay_i if max_Delay_i > 0 else 0)
    else:
        Score = 0

    # Output
    K = len(output_schedule)
    if K != sum_m_i:
        print(f"Error: Output packet number {K} does not match sum of input packet numbers {sum_m_i}", file=sys.stderr)
        sys.exit(1)
    print(K)
    for item in output_schedule:
        print(f"{item[0]} {item[1]} {item[2]}", end=' ')
    print()

if __name__ == "__main__":
    main()