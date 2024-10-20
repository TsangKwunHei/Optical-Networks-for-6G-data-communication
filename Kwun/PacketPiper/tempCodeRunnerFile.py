# Calculate the score based on the problem's scoring formula
    fi_sum = 0
    max_delay = 0
    for s in slices:
        fi = 1 if s['max_delay'] <= s['UBD_i'] else 0
        fi_sum += fi / n
        if s['max_delay'] > max_delay:
            max_delay = s['max_delay']
            max_delay_te = s 
    if max_delay > 0: