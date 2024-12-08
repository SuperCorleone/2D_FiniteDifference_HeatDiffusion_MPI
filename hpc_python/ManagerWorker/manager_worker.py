from mandelbrot_task import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI # MPI_Init and MPI_Finalize automatically called
import numpy as np
import sys
import time

# some parameters
MANAGER = 0       # rank of manager
TAG_TASK      = 1 # task       message tag
TAG_TASK_DONE = 2 # tasks done message tag
TAG_DONE      = 3 # done       message tag

def manager(comm, tasks):
    """
    The manager.

    Parameters
    ----------
    comm : mpi4py.MPI communicator
        MPI communicator
    tasks : list of objects with a do_task() method perfroming the task
        List of tasks to accomplish

    Returns
    -------
    ... ToDo ...
    """
    size = comm.Get_size()
    num_workers = size - 1  # Exclude the manager itself
    task_queue = tasks.copy()
    tasks_done = {worker: 0 for worker in range(1, size)}

    # Distribute initial tasks to workers
    for worker in range(1, size):
        if task_queue:
            task = task_queue.pop(0)
            comm.send(task, dest=worker, tag=TAG_TASK)
        else:
            comm.send(None, dest=worker, tag=TAG_DONE)  # No more tasks

    results = []

    # Collect results and send more tasks
    while len(results) < len(tasks):
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_TASK_DONE, status=status)
        results.append(result)

        worker = status.source
        tasks_done[worker] += 1

        # Assign a new task to the worker, if available
        if task_queue:
            task = task_queue.pop(0)
            comm.send(task, dest=worker, tag=TAG_TASK)
        else:
            comm.send(None, dest=worker, tag=TAG_DONE)  # Signal no more tasks

    # Collect remaining results from workers
    for worker in range(1, size):
        comm.send(None, dest=worker, tag=TAG_DONE)  # All workers should stop

    return results, tasks_done


def worker(comm):
    """
    The worker.

    Parameters
    ----------
    comm : mpi4py.MPI communicator
        MPI communicator
    """
    while True:
        task = comm.recv(source=MANAGER, tag=MPI.ANY_TAG, status=MPI.Status())
        tag = task.tag

        if tag == TAG_DONE:
            break  # No more tasks, exit
        elif tag == TAG_TASK:
            task.do_work()
            comm.send(task, dest=MANAGER, tag=TAG_TASK_DONE)


def readcmdline(rank):
    """
    Read command line arguments

    Parameters
    ----------
    rank : int
        Rank of calling MPI process

    Returns
    -------
    nx : int
        number of gridpoints in x-direction
    ny : int
        number of gridpoints in y-direction
    ntasks : int
        number of tasks
    """
    # report usage
    if len(sys.argv) != 4:
        if rank == MANAGER:
            print("Usage: manager_worker nx ny ntasks")
            print("  nx     number of gridpoints in x-direction")
            print("  ny     number of gridpoints in y-direction")
            print("  ntasks number of tasks")
        sys.exit()

    # read nx, ny, ntasks
    nx = int(sys.argv[1])
    if nx < 1:
        sys.exit("nx must be a positive integer")
    ny = int(sys.argv[2])
    if ny < 1:
        sys.exit("ny must be a positive integer")
    ntasks = int(sys.argv[3])
    if ntasks < 1:
        sys.exit("ntasks must be a positive integer")

    return nx, ny, ntasks


if __name__ == "__main__":

    # get COMMON WORLD communicator, size & rank
    comm    = MPI.COMM_WORLD
    size    = comm.Get_size()
    my_rank = comm.Get_rank()

    # report on MPI environment
    if my_rank == MANAGER:
        print(f"MPI initialized with {size:5d} processes")

    # read command line arguments
    nx, ny, ntasks = readcmdline(my_rank)

    # start timer
    timespent = - time.perf_counter()

    # trying out ... YOUR MANAGER-WORKER IMPLEMENTATION HERE ...
    x_min = -2.
    x_max  = +1.
    y_min  = -1.5
    y_max  = +1.5
    M = mandelbrot(x_min, x_max, nx, y_min, y_max, ny, ntasks)
    tasks = M.get_tasks()
    for task in tasks:
        task.do_work()
    m = M.combine_tasks(tasks)
    plt.imshow(m.T, cmap="gray", extent=[x_min, x_max, y_min, y_max])
    plt.savefig("mandelbrot.png")

    # stop timer
    timespent += time.perf_counter()

    # inform that done
    if my_rank == MANAGER:
        print(f"Run took {timespent:f} seconds")
        for i in range(size):
            if i == MANAGER:
                continue
            print(f"Process {i:5d} has done {TasksDoneByWorker[i]:10d} tasks")
        print("Done.")
