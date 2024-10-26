from dataclasses import dataclass
import functools


@dataclass
class Packet:
    id: int
    slice_id: int
    arrival_time: int
    size: int
    deadline: int


@dataclass
class Slice:
    packets: list[Packet]
    slice_bandwidth: float
    max_delay: int
    current_index: int = 0


def read_env():
    data = [n for n in input().split()]
    num_slices, port_bandwidth = int(data[0]), float(data[1])
    slices: list[Slice] = []
    for slice_id in range(num_slices):
        data = [n for n in input().split()]
        num_packets, slice_bandwidth, max_delay = (
            int(data[0]),
            float(data[1]),
            int(data[2]),
        )
        data = [int(n) for n in input().split()]
        packets: list[Packet] = []
        i = 0
        for packet_id in range(num_packets):
            packets.append(
                Packet(packet_id, slice_id, data[i], data[i + 1], data[i] + max_delay)
            )
            i += 2
        slices.append(Slice(packets, slice_bandwidth, max_delay))

    # Bandwidth is in Gbps
    return slices, port_bandwidth


def transmission_time(packet_size: int, port_bandwidth: float):
    return int(packet_size // port_bandwidth)


def earliest_packet(slices: list[Slice]) -> Packet:
    earliest_time: int | None = None
    earliest_slice: Slice | None = None
    for slice in slices:
        if not slice.packets:
            continue
        if slice.current_index >= len(slice.packets):
            continue
        packet = slice.packets[slice.current_index]
        if earliest_time is None or packet.arrival_time < earliest_time:
            earliest_time = packet.arrival_time
            earliest_slice = slice
    if earliest_slice is None:
        raise ValueError("No packets found")

    earliest_slice.current_index += 1
    return earliest_slice.packets[earliest_slice.current_index - 1]


def packets_before(slices: list[Slice], ts: int):
    packets: dict[int, list[Packet]] = {slice_id: [] for slice_id in range(len(slices))}
    for slice in slices:
        while slice.current_index < len(slice.packets):
            packet = slice.packets[slice.current_index]
            if packet.arrival_time < ts:
                packets[slice_id].append(packet)
                slice.current_index += 1
            else:
                break
    return packets


def schedule_packets(
    slices: list[Slice], port_bandwidth: float
) -> list[tuple[int, int, int]]:
    # current_time, slice_id, packet_id
    scheduled_packets: list[tuple[int, int, int]] = []

    slice_schedule: list[tuple[int, int]] = []

    current_time = 0

    available_packets: dict[int, list[Packet]] = {
        slice_id: [] for slice_id in range(len(slices))
    }

    total_num_packets = sum(len(slice.packets) for slice in slices)
    current_num_packets = 0

    while True:
        if len(slice_schedule) == total_num_packets:
            break

        if current_num_packets == 0:
            packet = earliest_packet(slices)
            available_packets[packet.slice_id].append(packet)
            current_num_packets += 1
        else:
            if not total_num_packets == current_num_packets:
                for slice_id, packets in packets_before(slices, current_time).items():
                    available_packets[slice_id].extend(packets)
                    current_num_packets += len(packets)

        # Sort the slices in available_packets by custom cmp function
        slices_to_schedule = [
            slice_id
            for slice_id in available_packets
            if len(available_packets[slice_id]) > 0
        ]

        def slice_sort(s1: int, s2: int):
            # deadline - time to transmit
            pkt1 = available_packets[s1][0]
            pkt2 = available_packets[s2][0]

            t1 = pkt1.deadline - transmission_time(pkt1.size, port_bandwidth)
            t2 = pkt2.deadline - transmission_time(pkt2.size, port_bandwidth)

            if t1 == t2:
                return pkt1.slice_id - pkt2.slice_id
            return t1 - t2

        slices_to_schedule.sort(key=functools.cmp_to_key(slice_sort))

        # Take the slice with the earliest deadline
        slice_id = slices_to_schedule[0]
        packet = available_packets[slice_id].pop(0)

        # Calculate the time it takes to transmit the packet

        slice_schedule.append((current_time, slice_id))
        current_time += transmission_time(packet.size, port_bandwidth)
        current_num_packets -= 1

    for ts, slice_id in slice_schedule:
        packet = slices[slice_id].packets.pop(0)
        scheduled_packets.append((ts, slice_id, packet.id))

    return scheduled_packets


def calculate_score(
    scheduled_packets: list[tuple[int, int, int]], slices: list[Slice]
) -> float:
    max_delays = [0] * len(slices)
    for leave_time, slice_id, packet_id in scheduled_packets:
        packet = slices[slice_id].packets[packet_id]
        delay = leave_time - packet.arrival_time
        if delay < 0:
            raise ValueError("Packet left before it arrived")
        max_delays[slice_id] = max(max_delays[slice_id], delay)

    slices_within_tolerance = sum(
        1 for i, delay in enumerate(max_delays) if delay <= slices[i].max_delay
    )
    overall_max_delay = max(max_delays)
    if overall_max_delay == 0:
        raise ValueError("Packet scheduling is invalid")

    return slices_within_tolerance / len(slices) + 10000 / overall_max_delay


if __name__ == "__main__":
    slices, bandwidth = read_env()
    scheduled_packets = schedule_packets(slices, bandwidth)

    print(len(scheduled_packets))
    for leave_time, slice_id, packet_id in scheduled_packets:
        print(f"{leave_time} {slice_id} {packet_id}", end=" ")
    print()
