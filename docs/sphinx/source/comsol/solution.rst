.. role:: raw-latex(raw)
   :format: latex
..

Calculating extracellular potentials with COMSOL
================================================

In order to calculate the extracellular potentials inside the tissue
that arise from electrical stimulation, we require a FEM model of tissue
on which the right boundary conditions are imposed. The whole process of
creating the model geometry, generating a mesh, assigning materials,
choosing physics, imposing boundary conditions… can be done in COMSOL.

Study types
-----------

The two most general study types in COMSOL are the Stationary study and
the Time Dependent study. As their names suggest, a stationary study
solves steady-state equations, while a time dependent study solves more
general, time-dependent equations over a certain duration of time.

1. One time-dependent study
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We are interested in both the spatial and the temporal behaviour of the
extracellular potentials in response to the imposed current injections.
While the potentials vary in time as a function of the injected current,
the quasi-static approximation allows us to consider the solution at
each point in time to be stationary. In the most general case, we need
to calculate the potentials for every time step using a Time Dependent
study. Although the final solution is time-dependent, it actually solves
time-independent equations in this case. We could represent the space-
and time-dependent solution :math:`V_{X_i,t_j}` in a matrix. There are
:math:`N` rows representing the FEM mesh nodes, and :math:`T` columns
representing the timestamps. Let’s call this matrix :math:`\bf{S}` for
solution.

.. math::

    \bf{S} = 
   \begin{bmatrix}
   V_{X_1,t_1} & V_{X_1,t_2} & \cdots & V_{X_1,t_T} \\
   V_{X_2,t_1} & V_{X_2,t_2} & \cdots & V_{X_2,t_T} \\
   \vdots      & \vdots      & \ddots & \vdots          \\
   V_{X_N,t_1} & V_{X_N,t_2} & \cdots & V_{X_N,t_T} \\
   \end{bmatrix}

This general case requires long COMSOL computations and offers little
flexibility when it comes to changing stimulation parameters (apart from
rerunning the COMSOL calculations). As a result, the two methods
described below are probably prefered over this one in most use cases.
Nevertheless, the output of a Time Dependent study in COMSOL can be
passed to BMTK.

2. One stationary study
~~~~~~~~~~~~~~~~~~~~~~~

Thanks to the quasi-static approximation, the FEM solution is linear
w.r.t. the injected current(s). In cases that are not too complex,
i.e. where the same current profile (but with possibly different
amplitudes) is used for all electrodes, the FEM solution only varies as
a function of the current profile, meaning the solutions at different
timestamps are linearly dependent. Such a matrix S is of rank 1 and can
be written as the outer product of two vectors.

.. math::

    \bf{S} = 
   \begin{bmatrix}
   V_{X_1,t_1} & V_{X_1,t_2} & \cdots & V_{X_1,t_T} \\
   V_{X_2,t_1} & V_{X_2,t_2} & \cdots & V_{X_2,t_T} \\
   \vdots      & \vdots      & \ddots & \vdots          \\
   V_{X_N,t_1} & V_{X_N,t_2} & \cdots & V_{X_N,t_T} \\
   \end{bmatrix}
   = \begin{bmatrix}
   V_{X_1}A_{t_1} & V_{X_1}A_{t_2} & \cdots & V_{X_1}A_{t_T} \\
   V_{X_2}A_{t_1} & V_{X_2}A_{t_2} & \cdots & V_{X_2}A_{t_T} \\
   \vdots      & \vdots      & \ddots & \vdots          \\
   V_{X_N}A_{t_1} & V_{X_N}A_{t_2} & \cdots & V_{X_N}A_{t_T} \\
   \end{bmatrix} 
   = \vec{V}_X \otimes \vec{A}_t

Here, the full solution can be described by the FEM solution at one time point (:math:`\vec{V_X}`) and a time-dependent scaling factor (i.e. the current profile :math:`\vec{A_t}`, which we will create in a next step). 
If possible, this is the easiest way to define extracellular potentials.

3. Multiple stationary studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because of the linearity of the solutions, the full solution
:math:`\bf{S}` can also be defined as the superposition (i.e. linear
combination) of the solutions :math:`\bf{S}_i` where each electrode is
active by itself.

.. math::  \bf{S} = \sum_i \bf{S}_i = \sum_i \vec{V}\_{X,i} {\otimes} \vec{A}\_{t,i} 

When only one electrode is active, the solution can always be decomposed
into a spatial component and a temporal component as in the paragraph
above. Doing this decomposition for each electrode separately and
linearly combining the solutions, only requires the FEM to be solved
once for each electrode. In most complex cases, this should be easier
than the first method.

Output
------

After a solution has been calculated, it can be exported with
Results>Export>Data.

-  File type: Text
-  Points to evaluate in: Take from dataset
-  Data format: Spreadsheet

This will generate a .txt file with a bunch of header rows (starting
with %), and then at least 4 space-separated columns. The first three
columns are the x-, y-, and -coordinate, where every row defines the
3D-coordinates of one of the mesh nodes.

Depending on whether simulation was stationary or time-dependent, there
will be either one or multiple extra columns. - Stationary: The 4th
column describes the potential at each point. This column is essentially
$ :raw-latex:`\vec{V_X}` $. - Time-dependent: Every column from the 4th
on contains the voltage profile at one timepoint T, similar to a vector
:math:`\begin{bmatrix} V_{X_0,t_T} & V_{X_1,t_T} & \cdots & V_{X_N,t_T}\end{bmatrix}^T`
that corresponds to a column of matrix :math:`S`.

Once the comsol.txt files have been obtained, they can be passed to
:ref:`BMTK`.
