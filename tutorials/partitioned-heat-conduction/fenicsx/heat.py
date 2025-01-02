"""
This code is mostly taken from: https://jsdokken.com/dolfinx-tutorial/chapter2/heat_equation.html
"""

import basix.ufl
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc, LinearProblem
from ufl import TrialFunction, TestFunction, inner, dx, grad, ds
import basix

import argparse
import numpy as np
from mpi4py import MPI
import sympy as sp

from fenicsxprecice import Adapter
from errorcomputation import compute_errors

from my_enums import ProblemType, DomainPart
from problem_setup import get_geometry

from dolfinx.io import VTXWriter

def determine_gradient(V_g, u):
    """
    compute flux following http://hplgit.github.io/INF5620/doc/pub/fenics_tutorial1.1/tu2.html#tut-poisson-gradu
    :param V_g: Vector function space
    :param u: solution where gradient is to be determined
    """

    w = TrialFunction(V_g)
    v = TestFunction(V_g)

    a = inner(w, v) * ufl.dx
    L = inner(grad(u), v) * ufl.dx
    problem = LinearProblem(a, L)
    return problem.solve()


# Parse arguments
parser = argparse.ArgumentParser(description="Solving heat equation for simple or complex interface case")
parser.add_argument("participantName", help="Name of the solver.", type=str, choices=[p.value for p in ProblemType])
parser.add_argument("-e", "--error-tol", help="set error tolerance", type=float, default=10**-8,)
args = parser.parse_args()
# Init variables with arguments
participant_name = args.participantName
error_tol = args.error_tol

t = 0
fenics_dt = 0.1
alpha = 3
beta = 1.2


# define the domain

if participant_name == ProblemType.DIRICHLET.value:
    problem = ProblemType.DIRICHLET
    domain_part = DomainPart.LEFT
elif participant_name == ProblemType.NEUMANN.value:
    problem = ProblemType.NEUMANN
    domain_part = DomainPart.RIGHT


# create domain and function space
domain, coupling_boundary, remaining_boundary = get_geometry(domain_part)
V = fem.functionspace(domain, ("Lagrange", 2))
element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(domain.geometry.dim,))
V_g = fem.functionspace(domain, element)
W, map_to_W = V_g.sub(0).collapse()

# Define the exact solution


class exact_solution():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t


u_exact = exact_solution(alpha, beta, t)

# Define the boundary condition
bcs = []
u_D = fem.Function(V)
u_D.interpolate(u_exact)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
# dofs for the coupling boundary
dofs_coupling = fem.locate_dofs_geometrical(V, coupling_boundary)
# dofs for the remaining boundary. Can be directly set to u_D
dofs_remaining = fem.locate_dofs_geometrical(V, remaining_boundary)
bc_D = fem.dirichletbc(u_D, dofs_remaining)
bcs.append(bc_D)


if problem is ProblemType.DIRICHLET:
    # Define flux in x direction
    f_N = fem.Function(W)
    f_N.interpolate(lambda x: 2 * x[0])


u_n = fem.Function(V)  # IV and solution u for the n-th time step
u_n.interpolate(u_exact)

# initialise precice
precice, precice_dt, initial_data = None, 0.0, None
if problem is ProblemType.DIRICHLET:
    precice = Adapter(adapter_config_filename="precice-adapter-config-D.json", mpi_comm=MPI.COMM_WORLD)
else:
    precice = Adapter(adapter_config_filename="precice-adapter-config-N.json", mpi_comm=MPI.COMM_WORLD)


if problem is ProblemType.DIRICHLET:
    precice.initialize(coupling_boundary, read_function_space=V, write_object=f_N)
elif problem is ProblemType.NEUMANN:
    precice.initialize(coupling_boundary, read_function_space=W, write_object=u_D)

# get precice's dt
precice_dt = precice.get_max_time_step_size()
dt = np.min([fenics_dt, precice_dt])


# Define the variational formualation

# As $f$ is a constant independent of $t$, we can define it as a constant.
f = fem.Constant(domain, beta - 2 - 2 * alpha)

# We can now create our variational formulation, with the bilinear form `a` and  linear form `L`.

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx

# create a coupling expression for the coupling_boundary and modify variational problem accordingly
coupling_expression = precice.create_coupling_expression()
if problem is ProblemType.DIRICHLET:
    # modify Dirichlet boundary condition on coupling interface
    bc_coup = fem.dirichletbc(coupling_expression, dofs_coupling)
    bcs.append(bc_coup)
if problem is ProblemType.NEUMANN:
    # modify Neumann boundary condition on coupling interface, modify weak
    # form correspondingly
    F += dt*coupling_expression * v * ufl.ds

a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

# ## Create the matrix and vector for the linear problem
# To ensure that we are solving the variational problem efficiently, we
# will create several structures which can reuse data, such as matrix
# sparisty patterns. Especially note as the bilinear form `a` is
# independent of time, we only need to assemble the matrix once.

A = assemble_matrix(a, bcs=bcs)
A.assemble()
b = create_vector(L)
uh = fem.Function(V)

# ## Define a linear variational solver
# We will use [PETSc](https://www.mcs.anl.gov/petsc/) to solve the
# resulting linear algebra problem. We use the Python-API `petsc4py` to
# define the solver. We will use a linear solver.

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

if problem is ProblemType.DIRICHLET:
    flux = fem.Function(V_g)

# boundaries point as always to the end of the timestep
u_exact.t += dt
u_D.interpolate(u_exact)

# create writer for output files
vtxwriter = VTXWriter(MPI.COMM_WORLD, f"output_{problem.name}.bp", [u_n])
vtxwriter.write(t)
    
while precice.is_coupling_ongoing():

    if precice.requires_writing_checkpoint():
        precice.store_checkpoint(u_n, t, 0)

    precice_dt = precice.get_max_time_step_size()
    dt = np.min([fenics_dt, precice_dt])

    read_data = precice.read_data(dt)

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, L)
    
    precice.update_coupling_expression(coupling_expression, read_data)

    # Apply Dirichlet boundary condition to the vector (according to the tutorial, the lifting operation is used to preserve the symmetry of the matrix)
    # As far as I understood, the boundary condition bc is updated by
    # u_D.interpolate above, since this function is wrapped into the bc object
    apply_lifting(b, [a], [bcs])
    set_bc(b, bcs)

    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)

    # Write data to preCICE according to which problem is being solved
    if problem is ProblemType.DIRICHLET:
        # Dirichlet problem reads temperature and writes flux on boundary to Neumann problem
        flux = determine_gradient(V_g, uh)
        flux_x = fem.Function(W)
        flux_x.interpolate(flux.sub(0))
        precice.write_data(flux_x)
        #precice.write_data(f_N)
    elif problem is ProblemType.NEUMANN:
        # Neumann problem reads flux and writes temperature on boundary to Dirichlet problem
        precice.write_data(uh)

    precice.advance(dt)
    precice_dt = precice.get_max_time_step_size()

    # roll back to checkpoint
    if precice.requires_reading_checkpoint():
        u_cp, t_cp, _ = precice.retrieve_checkpoint()
        u_n.x.array[:] = u_cp.x.array
        t = t_cp
    else:  # update solution
        # Update solution at previous time step (u_n)
        u_n.x.array[:] = uh.x.array
        t += float(dt)
        vtxwriter.write(t)

    if precice.is_time_window_complete():
        u_ref = fem.Function(V)
        u_ref.interpolate(u_D)
        error, error_pointwise = compute_errors(u_n, u_ref, total_error_tol=1)
        print("t = %.2f: L2 error on domain = %.3g" % (t, error))
        
        # Update Dirichlet BC
        u_exact.t += dt
        u_D.interpolate(u_exact)
        # TODO: update time dependent f (as soon as it is time dependent)!

precice.finalize()

vtxwriter.close()
