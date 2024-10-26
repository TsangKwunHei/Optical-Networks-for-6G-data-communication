import sys
import subprocess
import math
import statistics
import time
import os
import inspect

# Options to turn on/off each part
EVALUATE_SCORE = True          # Set to True or False to turn on/off Score evaluation
EVALUATE_CONSTRAINTS = True    # Set to True or False to turn on/off Constraints evaluation
EVALUATE_RULES = True          # Set to True or False to turn on/off Rules evaluation

# Option to use input_sim.txt as input stream instead of input_gen.py
USE_INPUT_TXT = True  # Set to True to use input.txt, False to use input_gen.py

def main():
    # Ensure no variable is named 'statistics' to prevent module shadowing
    import statistics  # Re-import to ensure it's not overshadowed

    # Determine the directory where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read input data
    if USE_INPUT_TXT:
        # Read input from input_sim.txt
        input_sim_path = os.path.join(script_dir, 'input.txt')
        try:
            with open(input_sim_path, 'r') as f:
                input_data = f.read()
        except Exception as e:
            print("Error reading input.txt:", e)
            sys.exit(1)
    else:
        # Read input from input_gen.py
        input_gen_path = os.path.join(script_dir, 'input_gen.py')
        try:
            input_gen_process = subprocess.Popen(
                [sys.executable, input_gen_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=script_dir  # Set current working directory
            )
            input_data, input_error = input_gen_process.communicate()
        except Exception as e:
            print("Error running input_gen.py:", e)
            sys.exit(1)

        if input_error:
            print("Error in input_gen.py execution:")
            print(input_error)
            sys.exit(1)

    # Optional: Print input_data for debugging
    # Uncomment the following line to inspect the input data
    # print("Input Data Passed to packet.py:", input_data)

    # Start timing for total runtime
    total_start_time = time.time()

    # Run packet.py with the input data
    packet_py_path = os.path.join(script_dir, 'GA.py')
    try:
        packet_process = subprocess.Popen(
            [sys.executable, packet_py_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=script_dir  # Set current working directory
        )
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
    try:
        num_slices = int(input_tokens[idx])
        idx += 1
        port_bw_gbps = float(input_tokens[idx])  # Port bandwidth in Gbps
        idx += 1
        port_bw = port_bw_gbps  # Keep port bandwidth in bits per nanosecond
    except (IndexError, ValueError) as e:
        print("Error parsing input data:", e)
        sys.exit(1)

    slices_info = []
    packet_map = {}
    for slice_id in range(num_slices):
        try:
            m_i = int(input_tokens[idx])
            idx += 1
            slice_bw_gbps = float(input_tokens[idx])
            idx += 1
            ubd_i = int(input_tokens[idx])
            idx += 1
        except (IndexError, ValueError) as e:
            print(f"Error parsing slice {slice_id} data:", e)
            sys.exit(1)

        packets = []
        for pkt_id in range(m_i):
            try:
                ts_ij = int(input_tokens[idx])
                idx += 1
                pkt_size = int(input_tokens[idx])
                idx += 1
            except (IndexError, ValueError) as e:
                print(f"Error parsing packet {pkt_id} in slice {slice_id}:", e)
                sys.exit(1)
            packet = {'ts': ts_ij, 'pkt_size': pkt_size, 'pkt_id': pkt_id, 'slice_id': slice_id}
            packets.append(packet)
            packet_map[(slice_id, pkt_id)] = packet

        slices_info.append({
            'm_i': m_i,
            'slice_bw': slice_bw_gbps,  # Keep slice bandwidth in bits per nanosecond (1 Gbps = 1 bit/ns)
            'ubd': ubd_i,
            'packets': packets,
            'slice_id': slice_id
        })

    # Parse output data from packet.py
    output_tokens = packet_output.strip().split()
    if not output_tokens:
        print("No output from packet.py")
        sys.exit(1)

    try:
        K = int(output_tokens[0])
    except ValueError:
        print("First token from packet.py output is not an integer (expected K)")
        sys.exit(1)

    output_schedule = []
    idx = 1
    while idx + 2 <= len(output_tokens):
        try:
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
        except ValueError as e:
            print("Error parsing packet schedule:", e)
            sys.exit(1)

    # Validate number of scheduled packets
    if K != len(output_schedule):
        print(f"Mismatch in number of scheduled packets: Expected {K}, Got {len(output_schedule)}")
        sys.exit(1)

    # Do NOT sort the output schedule by te
    # Preserve the original scheduling order as provided by the user's solution
    # output_schedule.sort(key=lambda x: x['te'])  # This line has been removed

    # Initialize data structures for evaluation
    fi_values = []
    delays = []
    delay_info = []
    constraint1_checks = []
    constraint2_checks = []
    constraint3_checks = []
    total_packets = 0

    # For Constraint 1 checks (Port bandwidth constraint)
    prev_te = None
    prev_pkt_size = None

    for scheduled_pkt in output_schedule:
        te_km = scheduled_pkt['te']
        slice_id = scheduled_pkt['slice_id']
        pkt_id = scheduled_pkt['pkt_id']

        # Get the packet info from packet_map
        key = (slice_id, pkt_id)
        if key not in packet_map:
            print(f"Packet not found in input data: Slice {slice_id}, Packet {pkt_id}")
            sys.exit(1)
        pkt_info = packet_map[key]
        pkt_size = pkt_info['pkt_size']

        if prev_te is not None and prev_pkt_size is not None:
            required_gap = prev_pkt_size / port_bw  # in nanoseconds (since port_bw is in bits/ns)
            actual_gap = te_km - prev_te
            meets_constraint = actual_gap >= required_gap
            percentage_exceed = ((actual_gap - required_gap) / required_gap) * 100 if required_gap > 0 else 0
            constraint1_checks.append({
                'te_i_j': prev_te,
                'te_k_m': te_km,
                'PktSize_i_j': prev_pkt_size,
                'PortBW': port_bw,
                'te_k_m_minus_te_i_j': actual_gap,
                'PktSize_div_PortBW': required_gap,
                'percentage_exceed': percentage_exceed,
                'meets_constraint': meets_constraint
            })
        prev_te = te_km
        prev_pkt_size = pkt_size

    # Process each slice individually
    for slice_info in slices_info:
        slice_id = slice_info['slice_id']
        scheduled_packets = [pkt for pkt in output_schedule if pkt['slice_id'] == slice_id]
        scheduled_packets.sort(key=lambda x: x['pkt_id'])  # Packets must be scheduled in order

        m_i = slice_info['m_i']
        if len(scheduled_packets) != m_i:
            print(f"Slice {slice_id}: Number of scheduled packets ({len(scheduled_packets)}) does not match input ({m_i}).")
            sys.exit(1)

        total_bits_sent = 0
        te_i_m = None
        ts_i_1 = None
        max_delay_i = 0

        previous_te = None

        for idx_p, scheduled_pkt in enumerate(scheduled_packets):
            pkt_id = scheduled_pkt['pkt_id']
            te_ij = scheduled_pkt['te']

            key = (slice_id, pkt_id)
            pkt_info = packet_map[key]
            ts_ij = pkt_info['ts']
            pkt_size = pkt_info['pkt_size']

            # Constraint 3: te_{i,j} >= ts_{i,j}
            meets_constraint_ts = te_ij >= ts_ij
            exceed_amount = te_ij - ts_ij
            constraint3_checks.append({
                'constraint': 'te_{i,j} >= ts_{i,j}',
                'slice_id': slice_id,
                'pkt_id': pkt_id,
                'te_ij': te_ij,
                'ts_ij': ts_ij,
                'exceed_amount': exceed_amount,
                'meets_constraint': meets_constraint_ts
            })

            # Constraint 3: te_{i,j+1} >= te_{i,j}
            if previous_te is not None:
                meets_constraint_te = te_ij >= previous_te
                gap = te_ij - previous_te
                constraint3_checks.append({
                    'constraint': 'te_{i,j+1} >= te_{i,j}',
                    'slice_id': slice_id,
                    'pkt_id': pkt_id,
                    'te_current': te_ij,
                    'te_previous': previous_te,
                    'gap': gap,
                    'meets_constraint': meets_constraint_te
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
        time_diff = te_i_m - ts_i_1  # in nanoseconds
        if time_diff > 0:
            actual_bw = total_bits_sent / time_diff  # bits per nanosecond
        else:
            actual_bw = float('inf')
        required_bw = 0.95 * slice_info['slice_bw']  # bits per nanosecond

        meets_constraint_bw = actual_bw >= required_bw
        percentage_exceed = ((actual_bw - required_bw) / required_bw) * 100 if required_bw > 0 else 0
        constraint2_checks.append({
            'slice_id': slice_id,
            'PacketSize_i': total_bits_sent,
            'te_i_m': te_i_m,
            'ts_i_1': ts_i_1,
            'PacketSize_i_div_time_diff': actual_bw,
            'SliceBW_i_times_0.95': required_bw,
            'Actual_BW': actual_bw,
            'percentage_exceed': percentage_exceed,
            'meets_constraint': meets_constraint_bw
        })

        # Calculate fi
        fi = 1 if max_delay_i <= slice_info['ubd'] else 0
        fi_values.append(fi)

        total_packets += m_i

    # Calculate score
    n = num_slices
    fi_sum = sum(fi_values) / n if n > 0 else 0
    max_delay = max(delays) if delays else 0
    if max_delay > 0:
        score = fi_sum + (10000 / max_delay)
    else:
        score = fi_sum + 10000  # Handle division by zero

    # Initialize constraint pass flag
    constraints_passed = True

    # Check Constraint 1: Port bandwidth constraint
    if EVALUATE_CONSTRAINTS:
        constraint1_met = all(check['meets_constraint'] for check in constraint1_checks)
        if not constraint1_met:
            constraints_passed = False
            print("\n=== Constraint 1 Violations (Port Bandwidth Constraint) ===")
            for check in constraint1_checks:
                if not check['meets_constraint']:
                    te_i_j = check['te_i_j']
                    te_k_m = check['te_k_m']
                    PktSize_i_j = check['PktSize_i_j']
                    PortBW = check['PortBW']
                    actual_gap = check['te_k_m_minus_te_i_j']
                    required_gap = check['PktSize_div_PortBW']
                    percentage_exceed = check['percentage_exceed']
                    print(f"te_km - te_i_j: {actual_gap} ns >= PktSize_i_j / PortBW: {required_gap:.6f} ns")
                    print(f"Actual gap exceeds required gap by {percentage_exceed:.2f}%\n")

    # Check Constraint 2: Output bandwidth constraint
    if EVALUATE_CONSTRAINTS:
        constraint2_met = all(check['meets_constraint'] for check in constraint2_checks)
        if not constraint2_met:
            constraints_passed = False
            print("\n=== Constraint 2 Violations (Output Bandwidth Constraint) ===")
            for check in constraint2_checks:
                if not check['meets_constraint']:
                    slice_id = check['slice_id']
                    packet_size_i = check['PacketSize_i']
                    te_i_m = check['te_i_m']
                    ts_i_1 = check['ts_i_1']
                    actual_bw = check['Actual_BW']
                    required_bw = check['SliceBW_i_times_0.95']
                    percentage_exceed = check['percentage_exceed']
                    print(f"Slice {slice_id}:")
                    print(f"  PacketSize_i / (te_i_m - ts_i_1): {packet_size_i} / {te_i_m - ts_i_1} = {actual_bw:.6f} bits/ns")
                    print(f"  Required BW: {required_bw:.6f} bits/ns (SliceBW_i * 0.95)")
                    print(f"  Actual BW meets required BW by {percentage_exceed:.2f}%\n")

    # Check Constraint 3: Packet scheduling sequence constraints
    if EVALUATE_CONSTRAINTS:
        constraint3_met = all(check['meets_constraint'] for check in constraint3_checks)
        if not constraint3_met:
            constraints_passed = False
            print("\n=== Constraint 3 Violations (Packet Scheduling Sequence Constraint) ===")
            for check in constraint3_checks:
                if not check['meets_constraint']:
                    if check['constraint'] == 'te_{i,j} >= ts_{i,j}':
                        slice_id = check['slice_id']
                        pkt_id = check['pkt_id']
                        te = check['te_ij']
                        ts = check['ts_ij']
                        exceed_amount = check['exceed_amount']
                        print(f"Slice {slice_id}, Packet {pkt_id}: te ({te}) < ts ({ts})")
                        print(f"  te does not meet ts by {exceed_amount} ns\n")
                    elif check['constraint'] == 'te_{i,j+1} >= te_{i,j}':
                        slice_id = check['slice_id']
                        pkt_id = check['pkt_id']
                        te_current = check['te_current']
                        te_previous = check['te_previous']
                        gap = check['gap']
                        print(f"Slice {slice_id}, Packet {pkt_id}: te_current ({te_current}) < te_previous ({te_previous})")
                        print(f"  te_current does not meet te_previous by {gap} ns\n")

    # **Important Correction: Set score to 0 if any constraints are violated**
    if EVALUATE_CONSTRAINTS:
        if not constraints_passed:
            score = 0
            print("\nConstraints not met. Test case score is 0.")
        else:
            print("\nAll constraints met.")

    # Calculate additional statistics for delays and fi
    if EVALUATE_SCORE:
        print("\n=== Score ===")
        print(f"Score: {score:.4f}")

        # Delay statistics
        print("\n=== Delay Statistics ===")
        if delays:
            max_delay_value = max(delays)
            max_delay_packet = max(delay_info, key=lambda x: x['delay'])
            avg_delay = sum(delays) / len(delays)
            median_delay = statistics.median(delays)
            std_delay = statistics.stdev(delays) if len(delays) > 1 else 0
            top_3_delays = sorted(delays, reverse=True)[:3]

            min_delay_value = min(delays)
            min_delay_packet = min(delay_info, key=lambda x: x['delay'])
            print(f"Max Delay: {max_delay_value} ns (Slice {max_delay_packet['slice_id']}, Packet {max_delay_packet['pkt_id']})")
            print(f"Min Delay: {min_delay_value} ns (Slice {min_delay_packet['slice_id']}, Packet {min_delay_packet['pkt_id']})")
            print(f"Average Delay: {avg_delay:.2f} ns")
            print(f"Median Delay: {median_delay:.2f} ns")
            print(f"Standard Deviation: {std_delay:.2f} ns")
            print(f"Top 3 Delays: {top_3_delays}")
        else:
            print("No delays recorded.")

        # fi statistics
        fi_percentage = (sum(fi_values) / len(fi_values)) * 100 if fi_values else 0
        print("\n=== fi Statistics ===")
        print(f"fi=1 for {sum(fi_values)} slices out of {n} ({fi_percentage:.2f}%)")
        print(f"Sum(fi / n): {fi_sum:.4f}")

    # Additional Constraint Statistics
    if EVALUATE_CONSTRAINTS:
        # Constraint 1 Percentages
        if constraint1_checks:
            percentages_exceed_c1 = [check['percentage_exceed'] for check in constraint1_checks if check['PktSize_div_PortBW'] > 0]
            if percentages_exceed_c1:
                try:
                    avg_pct_c1 = sum(percentages_exceed_c1) / len(percentages_exceed_c1)
                    median_pct_c1 = statistics.median(percentages_exceed_c1)
                    std_pct_c1 = statistics.stdev(percentages_exceed_c1) if len(percentages_exceed_c1) > 1 else 0
                    min_pct_c1 = min(percentages_exceed_c1)
                    print(inspect.cleandoc("""
                           Scheduling output sequence must meet the port bandwidth constraint (PortBW). 
                           The following requirements must be met:
                           te_{k,m} - te_{i,j} >= PktSize_{i,j} / PortBW

                           te_{i,j}        Leave time of previous packet
                           te_{k,m}        Leave time of next packet 
                           PktSize_{i,j}   Size of the processing Packet 
                           """))
                    print("\n--- Constraint 1 Statistics (% Exceedance) ---")
                    print(f"Average Percentage Exceedance: {avg_pct_c1:.2f}%")
                    print(f"Median Percentage Exceedance: {median_pct_c1:.2f}%")
                    print(f"Standard Deviation: {std_pct_c1:.2f}%")
                    print(f"Minimum Percentage Exceedance: {min_pct_c1:.2f}% (Closest to violating the constraint)")
                except Exception as e:
                    print(f"Error calculating Constraint 1 statistics: {e}")

        # Constraint 2 Percentages
        if constraint2_checks:
            percentages_exceed_c2 = [check['percentage_exceed'] for check in constraint2_checks if check['SliceBW_i_times_0.95'] > 0]
            if percentages_exceed_c2:
                try:
                    avg_pct_c2 = sum(percentages_exceed_c2) / len(percentages_exceed_c2)
                    median_pct_c2 = statistics.median(percentages_exceed_c2)
                    std_pct_c2 = statistics.stdev(percentages_exceed_c2) if len(percentages_exceed_c2) > 1 else 0
                    min_pct_c2 = min(percentages_exceed_c2)
                    print(inspect.cleandoc("""
                                Output bandwidth constraint for the ith slice (SliceBW_i)

                                Sum of PacketSize_i / (te_{i,m} - ts_{i,1}) >= 0.95 * SliceBW_i

                                PacketSize_i    Sum of all packet sizes in a slice
                                te_{i,m}        Departure time of last packet in the ith slice
                                ts_{i,1}        Arrival time of first packet in the ith slice 

                               """))
                    print("\n--- Constraint 2 Statistics (% Exceedance) ---")
                    print(f"Average Percentage Exceedance: {avg_pct_c2:.2f}%")
                    print(f"Median Percentage Exceedance: {median_pct_c2:.2f}%")
                    print(f"Standard Deviation: {std_pct_c2:.2f}%")
                    print(f"Minimum Percentage Exceedance: {min_pct_c2:.2f}% (Closest to violating the constraint)")
                except Exception as e:
                    print(f"Error calculating Constraint 2 statistics: {e}")

        # Constraint 3 Delays
        if delays:
            try:
                avg_delay = sum(delays) / len(delays)
                median_delay = statistics.median(delays)
                std_delay = statistics.stdev(delays) if len(delays) > 1 else 0
                min_delay = min(delays)
                min_delay_packet = min(delay_info, key=lambda x: x['delay'])
                print("\n--- Constraint 3 Delays (te_{i,j} - ts_{i,j}) ---")
                print(f"Average Delay: {avg_delay:.2f} ns")
                print(f"Median Delay: {median_delay:.2f} ns")
                print(f"Standard Deviation: {std_delay:.2f} ns")
                print(f"Minimum Delay: {min_delay} ns (Slice {min_delay_packet['slice_id']}, Packet {min_delay_packet['pkt_id']})")
            except Exception as e:
                print(f"Error calculating Constraint 3 delays: {e}")

            # Constraint 3 Gaps
            te_gaps = []
            te_gap_info = []
            for slice_info in slices_info:
                slice_id = slice_info['slice_id']
                scheduled_packets = [pkt for pkt in output_schedule if pkt['slice_id'] == slice_id]
                scheduled_packets.sort(key=lambda x: x['pkt_id'])
                for idx_p in range(1, len(scheduled_packets)):
                    te_current = scheduled_packets[idx_p]['te']
                    te_previous = scheduled_packets[idx_p -1]['te']
                    gap = te_current - te_previous
                    te_gaps.append(gap)
                    te_gap_info.append({
                        'slice_id': slice_id,
                        'pkt_id_current': scheduled_packets[idx_p]['pkt_id'],
                        'pkt_id_previous': scheduled_packets[idx_p -1]['pkt_id'],
                        'gap': gap
                    })

            if te_gaps:
                try:
                    avg_gap = sum(te_gaps) / len(te_gaps)
                    median_gap = statistics.median(te_gaps)
                    std_gap = statistics.stdev(te_gaps) if len(te_gaps) > 1 else 0
                    min_gap = min(te_gaps)
                    min_gap_info = min(te_gap_info, key=lambda x: x['gap'])

                    print("\n--- Constraint 3 Departure Time Gaps (te_{i,j+1} - te_{i,j}) ---")
                    print(f"Average Gap: {avg_gap:.2f} ns")
                    print(f"Median Gap: {median_gap:.2f} ns")
                    print(f"Standard Deviation: {std_gap:.2f} ns")
                    print(f"Minimum Gap: {min_gap} ns (Slice {min_gap_info['slice_id']}, Packets {min_gap_info['pkt_id_previous']} -> {min_gap_info['pkt_id_current']})")
                except Exception as e:
                    print(f"Error calculating Constraint 3 gaps: {e}")
            else:
                print("\nNo gaps recorded for Constraint 3.")

    # Print runtime information
    runtime_per_slice = total_runtime / n if n > 0 else 0
    runtime_per_packet = total_runtime / total_packets if total_packets > 0 else 0
    print("\n=== Runtime Information ===")
    print(f"Total Runtime: {total_runtime:.6f} seconds")
    print(f"Runtime per Slice: {runtime_per_slice:.6f} seconds (Total slices = {n})")
    print(f"Runtime per Packet: {runtime_per_packet:.6f} seconds (Total packets = {total_packets})")
    print("\n=== Score ===")
    print(f"Score: {score:.4f}")
if __name__ == "__main__":
    main()
