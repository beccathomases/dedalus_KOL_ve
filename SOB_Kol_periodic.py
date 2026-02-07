import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------
# Paper parameters / domain
# ------------------------
Lx = 10.0
y0, y1 = -1.0, 1.0     # paper uses y in [-1,1]
Nx, Ny = 256, 1024     # paper resolution (you can start smaller)

Re = 1e-2              # paper: Re = 1e-2
beta = 0.8             # choose (paper scans beta)
Wi = 30.0              # choose (paper scans Wi)
eps = 1e-3             # paper: epsilon = 1e-3
kappa = 5e-5           # paper: kappa = 5e-5

dealias = 3/2
timestepper = d3.RK222
dt_max = 5e-3
stop_sim_time = 5.0

dtype = np.float64

# ------------------------
# Bases: Fourier x, Chebyshev y
# ------------------------
coords = d3.CartesianCoordinates('x', 'y')
dist   = d3.Distributor(coords, dtype=dtype)

xb = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
yb = d3.ChebyshevT(coords['y'], size=Ny, bounds=(y0, y1), dealias=dealias)

x, y = dist.local_grids(xb, yb)
ex, ey = coords.unit_vector_fields(dist)

dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

# ------------------------
# Tau lift helper (1st-order tau method)
# ------------------------
lift_basis = yb.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

# ------------------------
# Fields: velocity, pressure, conformation components
# ------------------------
p  = dist.Field(name='p', bases=(xb, yb))
v  = dist.VectorField(coords, name='v', bases=(xb, yb))

cxx = dist.Field(name='cxx', bases=(xb, yb))
cxy = dist.Field(name='cxy', bases=(xb, yb))
cyy = dist.Field(name='cyy', bases=(xb, yb))

# Velocity taus (1st-order tau method)
tau_p  = dist.Field(name='tau_p')                  # no bases
tau_v1 = dist.VectorField(coords, name='tau_v1', bases=xb)
tau_v2 = dist.VectorField(coords, name='tau_v2', bases=xb)

# Conformation taus (per component)
tau_cxx1 = dist.Field(name='tau_cxx1', bases=xb)
tau_cxx2 = dist.Field(name='tau_cxx2', bases=xb)
tau_cxy1 = dist.Field(name='tau_cxy1', bases=xb)
tau_cxy2 = dist.Field(name='tau_cxy2', bases=xb)
tau_cyy1 = dist.Field(name='tau_cyy1', bases=xb)
tau_cyy2 = dist.Field(name='tau_cyy2', bases=xb)

# ------------------------
# First-order reductions
# ------------------------
grad_v  = d3.grad(v) + ey*lift(tau_v1)     # tau-corrected gradient for viscous + div-free
gradv   = grad_v

grad_cxx = d3.grad(cxx) + ey*lift(tau_cxx1)
grad_cxy = d3.grad(cxy) + ey*lift(tau_cxy1)
grad_cyy = d3.grad(cyy) + ey*lift(tau_cyy1)

# ------------------------
# Problem
# ------------------------
problem = d3.IVP(
    [p, v, cxx, cxy, cyy,
     tau_p, tau_v1, tau_v2,
     tau_cxx1, tau_cxx2, tau_cxy1, tau_cxy2, tau_cyy1, tau_cyy2],
    namespace=locals()
)

# Incompressibility with tau_p
problem.add_equation("trace(grad_v) + tau_p = 0")

# Momentum (paper eq. (2))
problem.add_equation(
    "dt(v) + grad(p) - (beta/Re)*div(grad_v) + lift(tau_v2)"
    " = -(v@grad(v))"
    "   + (1-beta)/(Re*Wi) * ( ex*(dx(cxx) + dy(cxy)) + ey*(dx(cxy) + dy(cyy)) )"
    "   + ex*(2/Re)"
)

# sPTT denominator (symbolic)
Trc = cxx + cyy
den = 1 - 2*eps + eps*Trc

# Conformation (paper eq. (1)) componentwise
problem.add_equation(
    "dt(cxx) - kappa*div(grad_cxx) + lift(tau_cxx2)"
    " = -(v@grad(cxx))"
    "   + 2*((gradv@ex)@ex)*cxx + 2*((gradv@ey)@ex)*cxy"
    "   - (cxx-1)/(Wi*den)"
)

problem.add_equation(
    "dt(cxy) - kappa*div(grad_cxy) + lift(tau_cxy2)"
    " = -(v@grad(cxy))"
    "   + ((gradv@ex)@ex)*cxy + ((gradv@ey)@ex)*cyy"
    "   + ((gradv@ex)@ey)*cxx + ((gradv@ey)@ey)*cxy"
    "   - (cxy)/(Wi*den)"
)

problem.add_equation(
    "dt(cyy) - kappa*div(grad_cyy) + lift(tau_cyy2)"
    " = -(v@grad(cyy))"
    "   + 2*((gradv@ex)@ey)*cxy + 2*((gradv@ey)@ey)*cyy"
    "   - (cyy-1)/(Wi*den)"
)

# ------------------------
# Boundary conditions
# ------------------------
problem.add_equation("v(y=-1) = 0")
problem.add_equation("v(y=+1) = 0")

# Starter conformation BCs: homogeneous Neumann
problem.add_equation("dy(cxx)(y=-1) = 0")
problem.add_equation("dy(cxx)(y=+1) = 0")
problem.add_equation("dy(cxy)(y=-1) = 0")
problem.add_equation("dy(cxy)(y=+1) = 0")
problem.add_equation("dy(cyy)(y=-1) = 0")
problem.add_equation("dy(cyy)(y=+1) = 0")

# Pressure gauge
problem.add_equation("integ(p) = 0")

# ------------------------
# Build solver
# ------------------------
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

def any_bad(*fields):
    """Return True if any field has NaN/Inf in grid space."""
    for f in fields:
        f.change_scales(1)
        g = f['g']
        if not np.isfinite(g).all():
            return True
    return False

# ------------------------
# Initial condition
# ------------------------
v['g'][0] = 1 - y**2
v['g'][1] = 0.0

cxx['g'] = 1.0
cxy['g'] = 0.0
cyy['g'] = 1.0

Delta = 0.01
cxx['g'] += Delta * np.exp(-(25/8)*((2*x/Lx - 0.5)**2 + y**2))

# ------------------------
# Output
# ------------------------
snap = solver.evaluator.add_file_handler('snapshots-morozov', sim_dt=0.1, max_writes=200)
snap.add_task(v@ex, name='vx')
snap.add_task(v@ey, name='vy')
snap.add_task(Trc, name='Trc')
snap.add_task(cxx, name='cxx')
snap.add_task(cxy, name='cxy')
snap.add_task(cyy, name='cyy')

vbar = d3.Integrate(v@ex, 'x') / Lx
omega = dx(v@ey) - dy((v@ex) - vbar)
snap.add_task(omega, name='omega_devmean')

# ------------------------
# Main loop (fixed dt, with NaN + Trc/den monitor)
# ------------------------
nan_cadence = 50
log_cadence = 20
dt = dt_max

logger.info("Starting main loop")
try:
    while solver.proceed:
        solver.step(dt)

        if solver.iteration % nan_cadence == 0:
            if any_bad(v, cxx, cxy, cyy):
                logger.error("NaN/Inf detected at iter=%d, t=%.6g. Stopping.",
                             solver.iteration, solver.sim_time)
                break

        if (solver.iteration - 1) % log_cadence == 0:
            cxx.change_scales(1)
            cyy.change_scales(1)
            Trc_g = cxx['g'] + cyy['g']
            den_g = (1 - 2*eps) + eps*Trc_g

            Trc_max = float(np.nanmax(Trc_g))
            den_min = float(np.nanmin(den_g))

            logger.info("iter=%d, t=%.4f, dt=%.2e | Trc_max=%.3e den_min=%.3e",
                        solver.iteration, solver.sim_time, dt, Trc_max, den_min)

            if den_min <= 0:
                logger.error("den_min <= 0 (%.3e) at iter=%d, t=%.6g. Stopping.",
                             den_min, solver.iteration, solver.sim_time)
                break

except Exception:
    logger.exception("Exception raised")
    raise
finally:
    try:
        solver.log_stats()
    except Exception:
        pass
