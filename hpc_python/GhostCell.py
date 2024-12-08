from mpi4py import MPI

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Step 1: Create 2D Cartesian topology
dims = MPI.Compute_dims(size, [0, 0])  # Automatically compute dimensions
periods = [1, 1]  # Enable periodic boundaries
reorder = True  # Allow process rank reordering for better performance

cart_comm = comm.Create_cart(dims, periods=periods, reorder=reorder)

# Step 2: Get Cartesian coordinates and neighboring ranks
coords = cart_comm.Get_coords(rank)
neighbors = {
    "East": cart_comm.Shift(0, 1)[1],  # +1 in x-direction
    "West": cart_comm.Shift(0, -1)[1],  # -1 in x-direction
    "North": cart_comm.Shift(1, 1)[1],  # +1 in y-direction
    "South": cart_comm.Shift(1, -1)[1],  # -1 in y-direction
}

# Step 3: Output the topology
print(f"Rank {rank}: Coords {coords}, Neighbors {neighbors}")

# Step 4: Perform ghost cell exchange
send_data = rank  # Data to send (rank itself)
recv_data = {
    "East": -1,
    "West": -1,
    "North": -1,
    "South": -1
}

# Send and receive ghost cell data
reqs = []
for direction, neighbor in neighbors.items():
    if neighbor != MPI.PROC_NULL:  # Valid neighbor
        if direction == "East":
            reqs.append(cart_comm.Isend(send_data, dest=neighbor))
            reqs.append(cart_comm.Irecv(recv_data["East"], source=neighbor))
        elif direction == "West":
            reqs.append(cart_comm.Isend(send_data, dest=neighbor))
            reqs.append(cart_comm.Irecv(recv_data["West"], source=neighbor))
        elif direction == "North":
            reqs.append(cart_comm.Isend(send_data, dest=neighbor))
            reqs.append(cart_comm.Irecv(recv_data["North"], source=neighbor))
        elif direction == "South":
            reqs.append(cart_comm.Isend(send_data, dest=neighbor))
            reqs.append(cart_comm.Irecv(recv_data["South"], source=neighbor))

# Wait for all communications to complete
MPI.Request.Waitall(reqs)

# Step 5: Output received data
print(f"Rank {rank} received: {recv_data}")
