import sys
import heapq

# Read input
n, PortBW = sys.stdin.readline().split()
n = int(n)
PortBW = float(PortBW) * 1e9  # Convert Gbps to bps

slices = []
for _ in range(n):
    m_i, SliceBW_i, UBD_i = sys.stdin.readline().split()
    m_i = int(m_i)
    SliceBW_i = float(SliceBW_i) * 1e9  # Convert Gbps to bps
    UBD_i = int(UBD_i)
    pkt_info = list(map(int, sys.stdin.readline().split()))
    ts_list = pkt_info[::2]
    size_list = pkt_info[1::2]
    packets = [{'ts': ts_list[i], 'size': size_list[i], 'id': i} for i in range(m_i)]
    slices.append({
        'packets': packets,
        'SliceBW_i': SliceBW_i,
        'UBD_i': UBD_i,
        'Delay_i': 0,
        'total_size': sum(size_list),
        'first_ts': ts_list[0],
        'last_te': ts_list[0],
        'scheduled_size': 0
    })

# Initialize scheduling variables
scheduled_packets = []
current_time = 0
port_available_time = 0
packet_heap = []

# Main scheduling loop
while any(s['packets'] for s in slices):
    # Add available packets to the heap
    for slice_idx, s in enumerate(slices):
        if s['packets']:
            pkt = s['packets'][0]
            if pkt['ts'] <= current_time:
                Delay_i_j = current_time - pkt['ts']
                weight = (Delay_i_j / s['UBD_i']) * (pkt['size'] / PortBW)
                heapq.heappush(packet_heap, (-weight, slice_idx, pkt))
    
    if not packet_heap:
        # Advance current_time to the next packet arrival
        next_ts = min(pkt['ts'] for s in slices if s['packets'] for pkt in [s['packets'][0]])
        current_time = max(current_time, next_ts)
        continue

    # Get the highest priority packet
    _, slice_idx, pkt = heapq.heappop(packet_heap)
    s = slices[slice_idx]

    # Ensure port bandwidth constraint
    te = max(current_time, port_available_time)
    te += pkt['size'] / PortBW  # Time to transmit packet

    # Update slice bandwidth usage
    s['scheduled_size'] += pkt['size']
    s['last_te'] = te

    # Update maximum delay for the slice
    Delay_i_j = te - pkt['ts']
    s['Delay_i'] = max(s['Delay_i'], Delay_i_j)

    # Schedule the packet
    scheduled_packets.append({
        'te': int(te),
        'SliceId': slice_idx,
        'PktId': pkt['id']
    })

    # Remove packet from slice
    s['packets'].pop(0)

    # Update times
    current_time = te
    port_available_time = te

# Validate constraints and compute f_i
for s in slices:
    duration = s['last_te'] - s['first_ts']
    if duration > 0:
        slice_bw_usage = s['scheduled_size'] / duration
        if slice_bw_usage < 0.95 * s['SliceBW_i']:
            print(0)  # Constraint violation
            sys.exit()
    else:
        print(0)  # Constraint violation
        sys.exit()
    s['f_i'] = 1 if s['Delay_i'] <= s['UBD_i'] else 0

# Output results
print(len(scheduled_packets))
output = []
for pkt in scheduled_packets:
    output.extend([pkt['te'], pkt['SliceId'], pkt['PktId']])
print(' '.join(map(str, output)))