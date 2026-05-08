# Active Nematic Hydrodynamics Simulator

A standalone browser toy for 2D active nematic hydrodynamics. Open `index.html`
in a browser and the simulation starts immediately.

There are two versions:

- `index.html`: finite-difference periodic-grid simulator.
- `fem.html`: finite-element-inspired simulator on a periodic triangular P1 mesh.
- `active_nematic.py`: Python reference simulator for command-line runs and
  data export.

## Run In The Browser

Open `index.html` directly, or use GitHub Pages once it is enabled for this
repository:

[https://asheshghosh.github.io/active-nematic-hydrodynamics/](https://asheshghosh.github.io/active-nematic-hydrodynamics/)

The FEM version is available at:

[https://asheshghosh.github.io/active-nematic-hydrodynamics/fem.html](https://asheshghosh.github.io/active-nematic-hydrodynamics/fem.html)

To enable Pages in GitHub: go to repository `Settings` -> `Pages`, set source to
`Deploy from a branch`, choose branch `main`, folder `/ (root)`, and save.

## Run In Python

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run a short simulation:

```bash
python3 active_nematic.py --grid 128 --steps 1000 --sample-every 10 --save-png
```

Outputs are written under `runs/active_nematic/`:

- `history.csv`: step, energy density, positive defects, negative defects.
- `final_state.npz`: final Q tensor, velocity, pressure, step, and energy.
- `final_snapshot.png`: optional summary plot if `matplotlib` is installed.

## What It Simulates

The state is a planar nematic $Q$ tensor:

$$
\mathbf Q =
\begin{pmatrix}
q_{xx} & q_{xy} \\
q_{xy} & -q_{xx}
\end{pmatrix}
$$

The molecular field is approximated as:

$$
\mathbf H =
K \nabla^2 \mathbf Q
+ a\left(S_0^2 - |\mathbf Q|^2\right)\mathbf Q
$$

The nematic tensor evolves according to:

$$
\partial_t \mathbf Q
+ \mathbf u \cdot \nabla \mathbf Q
=
\Gamma \mathbf H
+ [\mathbf \Omega, \mathbf Q]
+ \lambda \mathbf E
+ \boldsymbol\xi
$$

The velocity evolves with active stress forcing, viscosity, frictional damping,
and optional nonlinear momentum advection:

$$
\partial_t \mathbf u
+ \mathbf u \cdot \nabla \mathbf u
=
-\nabla p
+ \eta \nabla^2 \mathbf u
- \alpha \mathbf u
- \zeta \nabla \cdot \mathbf Q
$$

with incompressibility and periodic boundaries:

$$
\nabla \cdot \mathbf u = 0,
\qquad
\mathbf x \in \mathbb T^2
$$

The app evolves Q with:

- elastic relaxation from a one-constant Laplacian term,
- a soft bulk term that keeps nematic order near a target magnitude,
- flow alignment and local rotation from the velocity gradient,
- advection by the velocity field,
- active forcing from the active-stress divergence:

  $$
  -\zeta \nabla \cdot \mathbf Q
  $$
- viscous damping and pressure projection for an approximately incompressible
  velocity field,
- periodic boundaries.

This is an educational/interactable model, not a validated research solver.

## FEM Version

The FEM page, `fem.html`, uses a periodic triangular mesh with node-based
piecewise-linear fields. The implementation assembles a graph/cotangent-style
stiffness operator for Laplacian terms, uses lumped mass for explicit updates,
and estimates gradients from the nodal P1 field on the periodic mesh. It keeps
the same active nematic model and controls, but the spatial operators are
mesh-based rather than direct finite differences.

## Controls

- `Activity zeta`: positive and negative values favor different active stress
  responses.
- `Grid cells per side`: resets the periodic box with a coarser or finer
  square lattice, from $48 \times 48$ up to $1024 \times 1024$.
- `Active forcing`: toggles the $-\zeta \nabla \cdot \mathbf Q$ forcing term.
- `Extensile` / `Contractile`: chooses the sign of $\zeta$.
- `Activity magnitude`: controls the absolute strength of active forcing.
- `Elasticity K`: higher values suppress sharp director distortions.
- `Viscosity`: higher values smooth and slow the flow field.
- `Nonlinear advection`: toggles the inertial momentum term
  $\mathbf u \cdot \nabla \mathbf u$ in the velocity equation.
- `Friction alpha`: controls substrate/frictional damping $-\alpha \mathbf u$;
  set it to zero for an undamped wet-flow limit.
- `Flow alignment`: controls coupling between nematic order and strain.
- `Noise`: injects small fluctuations into Q.
- `Kick`: reseeds with a stronger perturbation.
- `Director render mode`: switches director visualization between rods,
  integrated streamlines, and a dense stencil texture.

The render shows vorticity/order as color, director rods as short white lines,
velocity as teal strokes, and detected half-charge defects as colored points.
The lower plot tracks energy density, positive defects, and negative defects
against simulation steps.
The pressure plot shows the instantaneous pressure-projection field with blue
for negative pressure, red for positive pressure, and the spatial mean removed.
