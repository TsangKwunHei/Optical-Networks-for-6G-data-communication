import random
import os


def generate_input():
    """Generates input data and writes it to a text file in the current script directory."""
    # Get the directory where the script is being run
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the full path to the output file in the same folder as the script
    file_path = os.path.join(script_dir, 'input_sim.txt')

    # Open the output file in the current directory of the script
    with open(file_path, 'w') as f_out:
        # Generate the number of slice users n (1 ≤ n ≤ 10000) and PortBW (1 Gbps to 800 Gbps)
        n = random.randint(1, 10000) #  (1 ≤ n ≤ 10000)
        port_bw_gps = random.randint(1, 800)  # From 1 Gbps to 800 Gbps.

        # Write Line 1: Number of slice users and PortBW
        line1 = f"{n} {port_bw_gps}"
        print(line1)
        f_out.write(line1 + '\n')

        for i in range(1, n + 1):
            # Generate m_i: number of packets in slice i (randomly chosen up to 1000)
            m_i = random.randint(1, 10000)
            # Generate SliceBW_i: slice bandwidth (0.01 Gbps to 10 Gbps) - This remains a decimal
            slice_bw_i = round(random.uniform(0.01, 10), 2)
            #slice_bw_i = random.randint(1, 10)
            # Generate UBD_i: maximum slice delay tolerance (random integer in nanoseconds)
            ubd_i = random.randint(20000, 40000)

            # Write Line 2i: m_i, SliceBW_i, UBD_i
            line2i = f"{m_i} {slice_bw_i} {ubd_i}"
            print(line2i)
            f_out.write(line2i + '\n')

            # Generate sequence information for slice i
            ts_list = []
            pkt_size_list = []
            ts_prev = 0
            for j in range(m_i):
                # Generate ts_{i,j}: arrival time of packet j (non-decreasing, integer)
                ts_j = ts_prev + random.randint(0, 20000)  # Increment up to 10000 ns
                ts_list.append(ts_j)
                ts_prev = ts_j
                # Generate PktSize_{i,j}: packet size (512 bits to 76800 bits, integer)
                pkt_size = random.randint(8000, 30000)
                pkt_size_list.append(pkt_size)

            # Prepare Line 2i+1: sequence information (ts and PktSize pairs)
            seq_info = ' '.join(f"{ts} {pkt_size}" for ts, pkt_size in zip(ts_list, pkt_size_list))
            print(seq_info)
            f_out.write(seq_info + '\n')


if __name__ == "__main__":
    generate_input()