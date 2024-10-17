import sys
import threading
import heapq

def main():
    import sys

    # Fast input reading
    data = sys.stdin.read().split()
    idx = 0

    # Read number of slices and PortBW
    n = int(data[idx]); idx += 1
    PortBW_Gbps = float(data[idx]); idx += 1  # in Gbps
    PortBW = int(PortBW_Gbps)  # bits/ns, converted to integer

    # Initialize slices data structures
    slices_packets = []            # List of lists: packets per slice
    slices_ptr = []                # List of ints: pointer to next packet per slice

    total_packets = 0

    for slice_id in range(n):
        m_i = int(data[idx]); idx += 1
        SliceBW_i = float(data[idx]); idx += 1  # in Gbps, not used directly in scheduling
        UBD_i = int(data[idx]); idx += 1         # in ns, used to calculate deadlines
        packets = []
        for pkt_id in range(m_i):
            ts = int(data[idx]); idx += 1        # arrival time in ns
            pkt_size = int(data[idx]); idx += 1  # in bits
            deadline = ts + UBD_i
            packets.append( (ts, pkt_size, pkt_id, slice_id, deadline) )
        # Sort packets by arrival time
        packets.sort()
        slices_packets.append(packets)
        slices_ptr.append(0)
        total_packets += m_i

    # Initialize heap with the first packet from each slice that has arrived by current_time=0
    heap = []
    for slice_id in range(n):
        if slices_ptr[slice_id] < len(slices_packets[slice_id]):
            pkt = slices_packets[slice_id][slices_ptr[slice_id]]
            ts, pkt_size, pkt_id, slice_id, deadline = pkt
            if ts <= 0:
                heapq.heappush(heap, (deadline, -slice_id, slice_id, pkt_id))
                slices_ptr[slice_id] += 1

    current_time = 0
    port_available_time = 0
    scheduled_packets = []

    while len(scheduled_packets) < total_packets:
        while heap:
            # Pop the packet with the earliest deadline
            deadline, neg_slice_id, slice_id, pkt_id = heapq.heappop(heap)
            pkt = slices_packets[slice_id][pkt_id]
            ts, pkt_size, pkt_id, slice_id, deadline = pkt

            # Calculate departure time te: max(current_time, ts, port_available_time)
            te = max(current_time, ts, port_available_time)

            # Calculate transmission time using integer arithmetic
            transmission_time = (pkt_size + PortBW -1) // PortBW

            # Calculate transmission end time
            te_end = te + transmission_time

            # Record the scheduled packet
            scheduled_packets.append( (te, slice_id, pkt_id) )

            # Update port availability and current_time
            port_available_time = te_end
            current_time = te_end

            # Push the next packet from the same slice into the heap if it has arrived by te_end
            if slices_ptr[slice_id] < len(slices_packets[slice_id]):
                next_pkt = slices_packets[slice_id][slices_ptr[slice_id]]
                next_ts, next_pkt_size, next_pkt_id, next_slice_id, next_deadline = next_pkt
                if next_ts <= te_end:
                    heapq.heappush(heap, (next_deadline, -next_slice_id, next_slice_id, next_pkt_id))
                    slices_ptr[slice_id] += 1
            # After scheduling this packet, continue scheduling available packets
        else:
            # Heap is empty, advance current_time to the next packet's arrival time
            next_ts = sys.maxsize
            next_slice_id = -1
            next_pkt_id = -1
            for slice_id in range(n):
                if slices_ptr[slice_id] < len(slices_packets[slice_id]):
                    pkt = slices_packets[slice_id][slices_ptr[slice_id]]
                    ts, pkt_size, pkt_id, slice_id, deadline = pkt
                    if ts < next_ts:
                        next_ts = ts
                        next_slice_id = slice_id
                        next_pkt_id = pkt_id
            if next_ts == sys.maxsize:
                break  # No more packets to schedule
            # Advance current_time to the next packet's ts or port_available_time
            current_time = max(next_ts, port_available_time)
            # Push all packets that have arrived by current_time into the heap
            for slice_id in range(n):
                while slices_ptr[slice_id] < len(slices_packets[slice_id]):
                    pkt = slices_packets[slice_id][slices_ptr[slice_id]]
                    ts, pkt_size, pkt_id, slice_id, deadline = pkt
                    if ts <= current_time:
                        heapq.heappush(heap, (deadline, -slice_id, slice_id, pkt_id))
                        slices_ptr[slice_id] += 1
                    else:
                        break

    # Output generation
    K = len(scheduled_packets)
    print(K)
    output = []
    for te, slice_id, pkt_id in scheduled_packets:
        output.append(f"{te} {slice_id} {pkt_id}")
    print(' '.join(output))

if __name__ == "__main__":
    threading.Thread(target=main).start()