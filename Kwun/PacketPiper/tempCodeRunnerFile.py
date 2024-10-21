     Packets in the same slice must leave in the order in which they arrived, 
            and the packet leaving time must be longer than the packet arrival time.
            
            te_{i,j+1} => te_{i,j} packets in same slice leave in same order as they arrive
            te_{i,j} => ts_{i,j} packet leave time must be greater than it's arrival time (ofc stupid)
