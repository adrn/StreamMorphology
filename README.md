stream-morphology
=================

Studying the morphology of tidal streams in triaxial potentials.

Frequency mapping
=================

Potential
---------

For this work, we will use a Triaxial NFW potential with axis ratios b/a=0.8, c/a=0.6. This potential is included in the potential registry in this package (`streammorpholoy.potential_regsitry`) with the key `'triaxial-nfw'`.

Setting up initial conditions
-----------------------------

We will now generate grids of initial conditions (ICs) to study the frequency diffusion rates of orbits in this potential. We will generate two types of grids: at constant energy started from a plane, e.g., the x-z plane (y velocity determined from the energy), and at constant energy started with zero velocity, e.g., on an equipotential surface. To generate the grids::

    python scripts/freqmap/make_grid.py -E -0.21 --potential=triaxial-NFW \
    --ic-func=tube_grid_xz --dx=0.5 --dz=0.5 -v

    python scripts/freqmap/make_grid.py -E -0.21 --potential=triaxial-NFW \
    --ic-func=box_grid --approx_num=4096 -v

These scripts produce IC grids and store them in `output/freqmap/triaxial-NFW/.../w0.npy`.

Running freqmap
---------------

We could now run `scripts/freqmap/freqmap.py` on these files, however, unless you have ~10 days to spare, I wouldn't recommend running this on a single core. We can use MPI to distribute the tasks, e.g.::

    mpiexec -n XX python scripts/freqmap/freqmap.py --path=output/freqmap/triaxial-NFW/.../ --mpi

Instead, you should probably submit a job on a cluster...See `mpi-submit` for example Torque jobs.
