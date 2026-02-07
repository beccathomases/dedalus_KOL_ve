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
Wi = 20.0              # choose (paper scans Wi)
eps = 1e-3             # paper: epsilon = 1e-3
kappa = 5e-5           # paper: kappa = 5e-5

dealias = 3/2
timestepper = d3.RK222          # you can swap later to a 3rd-order IMEX RK
dt_max = 5e-3                   # paper uses dt = 5e-3
stop_sim_time = 50.0

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

# Conformation taus (do per component)
tau_cxx1 = dist.Field(name='tau_cxx1', bases=xb)
tau_cxx2 = dist.Field(name='tau_cxx2', bases=xb)
tau_cxy1 = dist.Field(name='tau_cxy1', bases=xb)
tau_cxy2 = dist.Field(name='tau_cxy2', bases=xb)
tau_cyy1 = dist.Field(name='tau_cyy1', bases=xb)
tau_cyy2 = dist.Field(name='tau_cyy2', bases=xb)

# ------------------------
# First-order reductions
#   grad_v enters incompressibility + viscous term
#   grad_c?? enters diffusion term kappa*lap(c??)
# ------------------------
grad_v  = d3.grad(v) + ey*lift(tau_v1)
gradv = grad_v   # tau-corrected gradient of v


grad_cxx = d3.grad(cxx) + ey*lift(tau_cxx1)
grad_cxy = d3.grad(cxy) + ey*lift(tau_cxy1)
grad_cyy = d3.grad(cyy) + ey*lift(tau_cyy1)

# Helpful shorthand
Trc = cxx + cyy
den = (1 - 2*eps + eps*Trc)  # sPTT denominator

# Divergence of conformation tensor (vector)

div_c = dist.VectorField(coords, name='div_c', bases=(xb, yb))
div_c['g'][0] = 0.0
div_c['g'][1] = 0.0


# We'll define it symbolically in equations using derivatives:
# div_c = [dx cxx + dy cxy, dx cxy + dy cyy]

dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

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

# Momentum (paper eq. (2)): dt v + v·∇v = -∇p + (beta/Re)∇²v + (1-beta)/(Re*Wi) ∇·c + [2/Re, 0]
# Using first-order form: div(grad_v) instead of lap(v), plus lift(tau_v2)
problem.add_equation(
    "dt(v) + grad(p) - (beta/Re)*div(grad_v) + lift(tau_v2)"
    " = -(v@grad(v))"
    "   + (1-beta)/(Re*Wi) * ( ex*(dx(cxx) + dy(cxy)) + ey*(dx(cxy) + dy(cyy)) )"
    "   + ex*(2/Re)"
)


# Conformation (paper eq. (1)):
# dt(c) + v·∇c - (∇v)^T·c - c·(∇v) = kappa ∇²c - (c - I)/(Wi*den)
# In 2D, I has components (1,0;0,1).
# We write each component equation explicitly.

Trc = cxx + cyy
den = 1 - 2*eps + eps*Trc


# cxx
# cxx
problem.add_equation(
    "dt(cxx) - kappa*div(grad_cxx) + lift(tau_cxx2)"
    " = -(v@grad(cxx))"
    "   + 2*((gradv@ex)@ex)*cxx + 2*((gradv@ey)@ex)*cxy"
    "   - (cxx-1)/(Wi*den)"
)

# cxy
problem.add_equation(
    "dt(cxy) - kappa*div(grad_cxy) + lift(tau_cxy2)"
    " = -(v@grad(cxy))"
    "   + ((gradv@ex)@ex)*cxy + ((gradv@ey)@ex)*cyy"
    "   + ((gradv@ex)@ey)*cxx + ((gradv@ey)@ey)*cxy"
    "   - (cxy)/(Wi*den)"
)

# cyy
problem.add_equation(
    "dt(cyy) - kappa*div(grad_cyy) + lift(tau_cyy2)"
    " = -(v@grad(cyy))"
    "   + 2*((gradv@ex)@ey)*cxy + 2*((gradv@ey)@ey)*cyy"
    "   - (cyy-1)/(Wi*den)"
)


# ------------------------
# Boundary conditions
# ------------------------
# No-slip (paper): v(x, y=±1)=0
problem.add_equation("v(y=-1) = 0")
problem.add_equation("v(y=+1) = 0")

# Conformation BCs:
# Paper says: c(x,±1,t) set to values obtained from eq (1) with kappa=0. :contentReference[oaicite:1]{index=1}
# A practical *starter* that is stable: homogeneous Neumann for diffusion (no flux of conformation)
# (You can refine to the exact kappa=0 boundary prescription once the core solver is solid.)
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
# Initial condition (paper eq. (4): localized perturbation to cxx)
# ------------------------
# Start from "almost laminar": v_x = 1 - y^2 is the Newtonian Poiseuille profile in y∈[-1,1].
# (Paper’s velocity scale differs slightly due to shear thinning; this is a good seed.)
v['g'][0] = 1 - y**2
v['g'][1] = 0.0

# Conformation initial guess: identity
cxx['g'] = 1.0
cxy['g'] = 0.0
cyy['g'] = 1.0

# Localized bump in cxx (paper eq. (4) form)
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

# “paper-style” vorticity uses deviation from mean profile:
vbar = d3.Integrate(v@ex, 'x') / Lx
omega = dx(v@ey) - dy((v@ex) - vbar)
snap.add_task(omega, name='omega_devmean')

# ------------------------
# Main loop (fixed dt first; add CFL later)
# ------------------------
# ------------------------
# Main loop (fixed dt like paper)
# ------------------------

# ------------------------
# Main loop (fixed dt, with NaN monitor + simple Trc/den monitor)
# ------------------------

nan_cadence = 50   # check every 50 iterations
log_cadence = 20   # log every 20 iterations

dt = dt_max

logger.info("Starting main loop")
try:
    while solver.proceed:
        solver.step(dt)

        # ---- NaN/Inf monitor ----
        if solver.iteration % nan_cadence == 0:
            if any_bad(v, cxx, cxy, cyy):
                logger.error(
                    "NaN/Inf detected at iter=%d, t=%.6g. Stopping.",
                    solver.iteration, solver.sim_time
                )
                break

        # ---- Regular logging + denominator sanity ----
        if (solver.iteration - 1) % log_cadence == 0:
            # Make sure conformation components are on base grid
            cxx.change_scales(1)
            cyy.change_scales(1)

            Trc_g = cxx['g'] + cyy['g']                 # numpy array
            den_g = (1 - 2*eps) + eps*Trc_g             # numpy array

            Trc_max = float(np.nanmax(Trc_g))
            den_min = float(np.nanmin(den_g))

            logger.info(
                "iter=%d, t=%.4f, dt=%.2e | Trc_max=%.3e den_min=%.3e",
                solver.iteration, solver.sim_time, dt, Trc_max, den_min
            )

            # Optional hard stop if the sPTT denominator goes bad
            if den_min <= 0:
                logger.error(
                    "den_min <= 0 (%.3e) at iter=%d, t=%.6g. Stopping.",
                    den_min, solver.iteration, solver.sim_time
                )
                break

except Exception:
    logger.exception("Exception raised")
    raise
finally:
    # log_stats can fail if solver died before timings were initialized
    try:
        solver.log_stats()
    except Exception:
        pass
