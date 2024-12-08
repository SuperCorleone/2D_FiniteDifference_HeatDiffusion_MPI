import numpy as np

class mandelbrot():
    """
    Mandelbrot task creator and collector.

    Parameters
    ----------
    x_min : float
        Lower real coordinate of set (x-coordinate)
    x_max : float
        Upper real coordinate of set (x-coordinate)
    nx_global : int
        (Global) number of grid points for real coordinate (x-dir.) of set
    y_min : float
        Lower imaginary coordinate of set (y-coordinate)
    y_max : float
        Upper imaginary coordinate of set (y-coordinate)
    ny_global : int
        (Global) number of grid points for imaginary coordinate (y-dir.) of set
    ntasks : int
        Number of tasks/(sub)sets/patches
    """
    def __init__(self, x_min, x_max, nx_global,
                       y_min, y_max, ny_global, ntasks):
        self._x_min     = x_min
        self._x_max     = x_max
        self._nx_global = nx_global
        self._dx        = (x_max - x_min)/(nx_global - 1)
        self._y_min     = y_min
        self._y_max     = y_max
        self._ny_global = ny_global
        self._dy        = (y_max - y_min)/(ny_global - 1)
        self._ntasks    = ntasks

    def get_tasks(self):
        """
        Generate list of tasks by decomposing the x-direction into the
        desired number of tasks.
        """
        tasks = []
        dx = self._dx
        dy = self._dy
        for itask in range(self._ntasks):
            i_start  = itask*(self._nx_global//self._ntasks)
            if itask == self._ntasks - 1:
                nx_local = self._nx_global - i_start
            else:
                nx_local = self._nx_global//self._ntasks
            x_start  = self._x_min + i_start*dx
            j_start  = 0
            ny_local = self._ny_global
            y_start  = self._y_min
            tasks += [mandelbrot_patch(i_start, nx_local, x_start, dx,
                                       j_start, ny_local, y_start, dy, 100)]
        return tasks

    def combine_tasks(self, tasks):
        """
        Combine tasks generated by get_tasks.

        Parameters
        ----------
        tasks : list of mandelbrot_patches

        Returns
        -------
        mandelbrot_set : numpy.array of bools
           Mandelbrot set
        """
        mandelbrot_set = np.zeros((self._nx_global, self._ny_global),
                                  dtype=bool)
        for task in tasks:
            ibeg = task._i_start
            iend = task._i_start + task._nx_local
            mandelbrot_set[ibeg:iend,:] = task._patch[:,:]
        return mandelbrot_set

class mandelbrot_patch():
    """
    Class for computing and representing a Mandelbrot (sub)set/patch.

    Parameters
    ----------
    i_start : int
        global start index in x-direction of patch
    nx_local : int
        local size of patch in x-direction
    x_start : float
        start x coordinate of patch
    dx : float
        grid spacing in x-direction
    j_start : int
        global start index in y-direction of patch
    ny_local : int
        local size of patch in y-direction
    y_start : float
        start y coordinate of patch
    dy : float
        grid spacing in y-direction
    Nmax : int
        Maximum number of iterations
    """
    def __init__(self, i_start, nx_local, x_start, dx,
                       j_start, ny_local, y_start, dy, Nmax):
        self._i_start  = i_start
        self._nx_local = nx_local
        self._x_start  = x_start
        self._dx       = dx
        self._j_start  = j_start
        self._ny_local = ny_local
        self._y_start  = y_start
        self._dy       = dy
        self._Nmax = Nmax

    def do_work(self):
        """Compute Mandelbrot (sub)set/patch."""
        x = self._x_start + np.arange(self._nx_local)*self._dx
        y = self._y_start + np.arange(self._ny_local)*self._dy
        c = x[:, np.newaxis] + 1j*y[np.newaxis, :]
        z = np.zeros_like(c)
        mask = np.full_like(c, fill_value=True, dtype=bool)
        z[mask] = c[mask]
        for k in range(self._Nmax):
            z[mask] = z[mask]**2 + c[mask]
            mask[np.abs(z) > 2.] = False
        self._patch = (np.abs(z) <= 2.)

def main():
    import matplotlib.pyplot as plt
    x_min  = -2.
    x_max  = +1.
    nx     = 201
    y_min  = -1.5
    y_max  = +1.5
    ny     = 301
    ntasks = 33
    M = mandelbrot(x_min, x_max, nx, y_min, y_max, ny, ntasks)
    tasks = M.get_tasks()
    for task in tasks:
        task.do_work()
    m = M.combine_tasks(tasks)
    plt.imshow(m.T, cmap="gray", extent=[x_min, x_max, y_min, y_max])
    plt.show()

if __name__ == "__main__":
    main()
