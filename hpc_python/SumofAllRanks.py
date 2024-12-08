from mpi4py import MPI
import numpy as np

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 1. Using pickle-based communication methods
def sum_ranks_pickle_based():
    # Rank as a generic Python object
    rank_value = rank
    # Perform allreduce to compute the sum of all ranks
    total_sum = comm.allreduce(rank_value, op=MPI.SUM)
    return total_sum

# 2. Using direct buffer-based communication methods
def sum_ranks_buffer_based():
    # Rank as a NumPy array
    rank_array = np.array(rank, dtype='i')  # 'i' for integer
    total_sum_array = np.array(0, dtype='i')  # Buffer to hold the result
    # Perform Allreduce to compute the sum of all ranks
    comm.Allreduce(rank_array, total_sum_array, op=MPI.SUM)
    return total_sum_array[0]

if __name__ == "__main__":
    # Compute sum using pickle-based communication
    total_sum_pickle = sum_ranks_pickle_based()
    if rank == 0:  # Print result from rank 0
        print(f"[Pickle-based] Sum of all ranks: {total_sum_pickle}")

    # Compute sum using buffer-based communication
    total_sum_buffer = sum_ranks_buffer_based()
    if rank == 0:  # Print result from rank 0
        print(f"[Buffer-based] Sum of all ranks: {total_sum_buffer}")
