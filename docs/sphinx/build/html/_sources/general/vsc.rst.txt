VSC
===

Getting proper mpi4py environment
---------------------------------

conda create -n neural -c intel intelpython3_full conda activate neural

Testing mpiexec -n 5 python -m mpi4py.bench helloworld

Getting BMTK and NEURON
-----------------------

conda install -c kaeldai bmtk pip3 install neuron # Testing Make a
python file called test.py

.. code:: python

   from mpi4py import MPI
   from neuron import h
   pc = h.ParallelContext()
   id = int(pc.id())
   nhost = int(pc.nhost())
   print("I am {} of {}".format(id, nhost))

and run it

``$ mpirun -n 4 python test.py``
