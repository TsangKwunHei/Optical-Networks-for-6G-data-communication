import subprocess
import sys
import math
import statistics
import time

def main():
    # Read input from input_gen.py
    try:
        input_gen_process = subprocess.Popen(['python', 'input_gen.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        input_data, input_error = input_gen_process.communicate()
    except Exception as e:
        print("Error running input_gen.py:", e)
        sys.exit(1)

    if input_error:
        print("Error in input_gen.py execution:")
        print(input_error)
        sys.exit(1)

    # Start timing for total runtime
    total_start_time = time.time()

    # Run packet.py with the input data
    try:
        packet_process = subprocess.Popen(['python', 'packet.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        packet_output, packet_error = packet_process.communicate(input=input_data)
    except Exception as e:
        print("Error running packet.py:", e)
        sys.exit(1)

    # End timing for total runtime
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time

    if packet_error:
        print("Error in packet.py execution:")
        print(packet_error)
        sys.exit(1)

    # Parse input data
    input_tokens = input_data.strip().split()
    idx = 0
    num_slices = int(input_tokens[idx])
    idx += 1
    port_bw_gbps = float(input_tokens[idx])
    idx += 1
    port_bw = port_bw_gbps * 1e9  # Convert Gbps to bps

    slices_info = []
    for slice_id in range(num_slices):
        m_i = int(input_tokens[idx])
        idx += 1
        slice_bw_gbps = float(input_tokens[idx])
        idx += 1
        ubd_i = int(input_tokens[idx])
        idx += 1

        packets = []
        for pkt_id in range(m_i):
            ts_ij = int(input_tokens[idx])
            idx += 1
            pkt_size = int(input_tokens[idx])
            idx += 1
            packets.append({'ts': ts_ij, 'pkt_size': pkt_size, 'pkt_id': pkt_id, 'slice_id': slice_id})

        slices_info.append({
            'm_i': m_i,
            'slice_bw': slice_bw_gbps * 1e9,  # Convert Gbps to bps
            'ubd': ubd_i,
            'packets': packets,
            'slice_id': slice_id
        })

    # Parse output data from packet.py
    output_tokens = packet_output.strip().split()
    if not output_tokens:
        print("No output from packet.py")
        sys.exit(1)

    K = int(output_tokens[0])
    output_schedule = []
    idx = 1
    while idx + 2 <= len(output_tokens):
        te = int(output_tokens[idx])
        idx += 1
        slice_id = int(output_tokens[idx])
        idx += 1
        pkt_id = int(output_tokens[idx])
        idx += 1
        output_schedule.append({
            'te': te,
            'slice_id': slice_id,
            'pkt_id': pkt_id
        })

    # Validate number of scheduled packets
    if K != len(output_schedule):
        print("Mismatch in number of scheduled packets")
        sys.exit(1)

    # Initialize data structures for evaluation
    fi_values = []
    delays = []
    delay_info = []
    constraint1_checks = []
    constraint2_checks = []
    constraint3_checks = []
    total_packets = 0

    # For Constraint 1 checks
    prev_te = None
    prev_pkt_size = None

    # Constraint 1: Port bandwidth constraint
    for idx, scheduled_pkt in enumerate(output_schedule):
        te_km = scheduled_pkt['te']
        slice_id = scheduled_pkt['slice_id']
        pkt_id = scheduled_pkt['pkt_id']

        pkt_info = next((pkt for pkt in slices_info[slice_id]['packets'] if pkt['pkt_id'] == pkt_id), None)
        if pkt_info is None:
            print(f"Packet not found in input data: Slice {slice_id}, Packet {pkt_id}")
            sys.exit(1)

        pkt_size = pkt_info['pkt_size']

        if prev_te is not None and prev_pkt_size is not None:
            required_gap = math.ceil(prev_pkt_size / port_bw * 1e9)  # Convert to ns
            actual_gap = te_km - prev_te
            constraint1_checks.append({
                'actual_gap': actual_gap,
                'required_gap': required_gap,
                'meets_constraint': actual_gap >= required_gap,
                'prev_te': prev_te,
                'te_km': te_km
            })
        prev_te = te_km
        prev_pkt_size = pkt_size

    # Constraint 2 and 3: Per-slice checks
    for slice_info in slices_info:
        slice_id = slice_info['slice_id']
        scheduled_packets = [pkt for pkt in output_schedule if pkt['slice_id'] == slice_id]
        scheduled_packets.sort(key=lambda x: x['pkt_id'])

        m_i = slice_info['m_i']
        if len(scheduled_packets) != m_i:
            print(f"Slice {slice_id}: Number of scheduled packets does not match input.")
            sys.exit(1)

        total_bits_sent = 0
        te_i_m = None
        ts_i_1 = None
        max_delay_i = 0

        previous_te = None

        for idx, scheduled_pkt in enumerate(scheduled_packets):
            pkt_id = scheduled_pkt['pkt_id']
            te_ij = scheduled_pkt['te']

            pkt_info = slice_info['packets'][pkt_id]
            ts_ij = pkt_info['ts']
            pkt_size = pkt_info['pkt_size']

            # Constraint 3: te_{i,j} >= ts_{i,j}
            if te_ij < ts_ij:
                constraint3_checks.append({
                    'slice_id': slice_id,
                    'pkt_id': pkt_id,
                    'te': te_ij,
                    'ts': ts_ij,
                    'meets_constraint': False
                })
            else:
                constraint3_checks.append({
                    'slice_id': slice_id,
                    'pkt_id': pkt_id,
                    'te': te_ij,
                    'ts': ts_ij,
                    'meets_constraint': True
                })

            # Constraint 3: te_{i,j+1} >= te_{i,j}
            if previous_te is not None:
                if te_ij < previous_te:
                    constraint3_checks.append({
                        'slice_id': slice_id,
                        'pkt_id': pkt_id,
                        'te': te_ij,
                        'prev_te': previous_te,
                        'meets_constraint': False
                    })
                else:
                    constraint3_checks.append({
                        'slice_id': slice_id,
                        'pkt_id': pkt_id,
                        'te': te_ij,
                        'prev_te': previous_te,
                        'meets_constraint': True
                    })
            previous_te = te_ij

            # Calculate delay
            delay_ij = te_ij - ts_ij
            if delay_ij > max_delay_i:
                max_delay_i = delay_ij

            delays.append(delay_ij)
            delay_info.append({
                'slice_id': slice_id,
                'pkt_id': pkt_id,
                'delay': delay_ij
            })

            total_bits_sent += pkt_size

            if ts_i_1 is None or ts_ij < ts_i_1:
                ts_i_1 = ts_ij

            if te_i_m is None or te_ij > te_i_m:
                te_i_m = te_ij

        # Constraint 2: Output bandwidth constraint
        if te_i_m != ts_i_1:
            actual_bw = total_bits_sent / (te_i_m - ts_i_1)  # in bits/ns
        else:
            actual_bw = float('inf')
        required_bw = 0.95 * slice_info['slice_bw'] / 1e9  # Convert to bits/ns

        constraint2_checks.append({
            'slice_id': slice_id,
            'actual_bw': actual_bw,
            'required_bw': required_bw,
            'meets_constraint': actual_bw >= required_bw
        })

        # Calculate fi
        if max_delay_i <= slice_info['ubd']:
            fi_values.append(1)
        else:
            fi_values.append(0)

        total_packets += m_i

    # Calculate score
    n = num_slices
    fi_sum = sum(fi_values) / n
    max_delay = max(delays) if delays else 0
    if max_delay > 0:
        score = fi_sum + (10000 / max_delay)
    else:
        score = fi_sum + 10000

    # Print Score and delay statistics
    print(f"Score: {score:.4f}")
    print("\nDelay Statistics:")
    if delays:
        max_delay_value = max(delays)
        max_delay_info = max(delay_info, key=lambda x: x['delay'])
        avg_delay = sum(delays) / len(delays)
        median_delay = statistics.median(delays)
        std_delay = statistics.stdev(delays) if len(delays) > 1 else 0
        top_3_delays = sorted(delays, reverse=True)[:3]

        print(f"Max Delay: {max_delay_value} ns (Slice {max_delay_info['slice_id']}, Packet {max_delay_info['pkt_id']})")
        print(f"Average Delay: {avg_delay:.2f} ns ({(avg_delay / max_delay_value) * 100:.2f}% of max)")
        print(f"Median Delay: {median_delay:.2f} ns ({(median_delay / max_delay_value) * 100:.2f}% of max)")
        print(f"Top 3 Delays: {top_3_delays}")
        print(f"Standard Deviation: {std_delay:.2f} ns")
    else:
        print("No delays recorded.")

    # Print fi statistics
    fi_percentage = (sum(fi_values) / len(fi_values)) * 100
    print(f"\nfi Statistics:")
    print(f"fi=1 for {sum(fi_values)} slices out of {n} ({fi_percentage:.2f}%)")
    print(f"Sum(fi / n): {fi_sum:.4f}")

    # Constraint checks
    constraints_passed = True

    # Constraint 1 Checks
    constraint1_met = all(check['meets_constraint'] for check in constraint1_checks)
    if not constraint1_met:
        constraints_passed = False
        print("\nConstraint 1 Violations (Port Bandwidth Constraint):")
        for check in constraint1_checks:
            if not check['meets_constraint']:
                print(f"Required gap: {check['required_gap']} ns, Actual gap: {check['actual_gap']} ns between te={check['prev_te']} and te={check['te_km']}")
    else:
        print("\nConstraint 1 met for all packets.")

    # Constraint 2 Checks
    constraint2_met = all(check['meets_constraint'] for check in constraint2_checks)
    if not constraint2_met:
        constraints_passed = False
        print("\nConstraint 2 Violations (Output Bandwidth Constraint):")
        for check in constraint2_checks:
            if not check['meets_constraint']:
                actual_bw_gbps = check['actual_bw'] * 1e9  # Convert to Gbps
                required_bw_gbps = check['required_bw'] * 1e9  # Convert to Gbps
                print(f"Slice {check['slice_id']}: Actual BW = {actual_bw_gbps:.6f} Gbps, Required BW = {required_bw_gbps:.6f} Gbps")
    else:
        print("\nConstraint 2 met for all slices.")

    # Constraint 3 Checks
    constraint3_met = all(check['meets_constraint'] for check in constraint3_checks)
    if not constraint3_met:
        constraints_passed = False
        print("\nConstraint 3 Violations (Packet Scheduling Sequence Constraint):")
        for check in constraint3_checks:
            if not check['meets_constraint']:
                if 'prev_te' in check:
                    print(f"Slice {check['slice_id']}, Packet {check['pkt_id']}: te_{check['pkt_id']} ({check['te']}) < te_{check['pkt_id'] - 1} ({check['prev_te']})")
                else:
                    print(f"Slice {check['slice_id']}, Packet {check['pkt_id']}: te ({check['te']}) < ts ({check['ts']})")
    else:
        print("\nConstraint 3 met for all packets.")

    # Print runtime information
    runtime_per_slice = total_runtime / n if n > 0 else 0
    runtime_per_packet = total_runtime / total_packets if total_packets > 0 else 0
    print(f"\nRuntime Information:")
    print(f"Total Runtime: {total_runtime:.6f} seconds")
    print(f"Runtime per Slice: {runtime_per_slice:.6f} seconds")
    print(f"Runtime per Packet: {runtime_per_packet:.6f} seconds")

    # Final check
    if not constraints_passed:
        print("\nConstraints not met. Test case score is 0.")
    else:
        print("\nAll constraints met.")

if __name__ == "__main__":
    main()
