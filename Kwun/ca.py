import cProfile
import pstats

def main():
    n, port_bw, slices = parse_input()
    ga_scheduler = packetSchedulingTask(slices, port_bw)
    best_schedule, best_obj = ga_scheduler.optimizeSchedule()
    ga_scheduler.outputResult()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)  # Print top 10 functions by cumulative time
