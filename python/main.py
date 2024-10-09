# main.py

def main():
    from input_processing import read_input
    from determine_bottleneck import determine_bottleneck
    from scenario_processing import process_scenarios

    n, m, ci, graph, services, edges = read_input()
    e_failed_list = determine_bottleneck(n, m, ci, graph, services, edges)
    process_scenarios(n, m, ci, graph, services, e_failed_list)

if __name__ == "__main__":
    main()