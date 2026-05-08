#!/usr/bin/env python3
"""Reference active nematic hydrodynamics simulator.

This is a small NumPy implementation of the same educational model used by the
browser app. It evolves a 2D periodic Q-tensor field, pressure-projects the
velocity, tracks energy and signed defects, and writes data products that are
easy to inspect or post-process.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Params:
    grid: int = 128
    steps: int = 1000
    dt: float = 0.18
    activity: float = 0.70
    activity_sign: float = 1.0
    active: bool = True
    elastic: float = 0.23
    viscosity: float = 0.24
    friction: float = 0.055
    alignment: float = 0.62
    noise: float = 0.09
    gamma: float = 0.68
    target_s: float = 0.66
    nonlinear_momentum: bool = True
    pressure_iterations: int = 22
    sample_every: int = 10
    seed: int = 4
    output_dir: Path = Path("runs/active_nematic")
    save_png: bool = False


class ActiveNematic:
    def __init__(self, params: Params) -> None:
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        n = params.grid
        x = np.arange(n)[None, :]
        y = np.arange(n)[:, None]
        wave = 0.35 * np.sin(2 * np.pi * x / n) * np.cos(2 * np.pi * y / n)
        theta = wave + 0.18 * self.rng.standard_normal((n, n))
        order = 0.62 + 0.05 * self.rng.standard_normal((n, n))
        self.qxx = order * np.cos(2 * theta)
        self.qxy = order * np.sin(2 * theta)
        self.ux = np.zeros((n, n), dtype=np.float64)
        self.uy = np.zeros((n, n), dtype=np.float64)
        self.pressure = np.zeros((n, n), dtype=np.float64)
        self.step_count = 0
        self.energy = 0.0

    @staticmethod
    def ddx(a: np.ndarray) -> np.ndarray:
        return 0.5 * (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1))

    @staticmethod
    def ddy(a: np.ndarray) -> np.ndarray:
        return 0.5 * (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0))

    @staticmethod
    def lap(a: np.ndarray) -> np.ndarray:
        return (
            np.roll(a, -1, axis=1)
            + np.roll(a, 1, axis=1)
            + np.roll(a, -1, axis=0)
            + np.roll(a, 1, axis=0)
            - 4 * a
        )

    def advect(self, a: np.ndarray) -> np.ndarray:
        return self.ux * self.ddx(a) + self.uy * self.ddy(a)

    def project_velocity(self, next_ux: np.ndarray, next_uy: np.ndarray) -> None:
        div = self.ddx(next_ux) + self.ddy(next_uy)
        self.pressure *= 0.82
        for _ in range(self.p.pressure_iterations):
            self.pressure = 0.25 * (
                np.roll(self.pressure, -1, axis=1)
                + np.roll(self.pressure, 1, axis=1)
                + np.roll(self.pressure, -1, axis=0)
                + np.roll(self.pressure, 1, axis=0)
                - div
            )
        self.ux = next_ux - self.ddx(self.pressure)
        self.uy = next_uy - self.ddy(self.pressure)

    def step(self) -> None:
        zeta = self.p.activity_sign * self.p.activity if self.p.active else 0.0
        force_x = -zeta * (self.ddx(self.qxx) + self.ddy(self.qxy))
        force_y = -zeta * (self.ddx(self.qxy) - self.ddy(self.qxx))
        adv_x = self.advect(self.ux) if self.p.nonlinear_momentum else 0.0
        adv_y = self.advect(self.uy) if self.p.nonlinear_momentum else 0.0
        next_ux = self.ux + self.p.dt * (
            force_x + self.p.viscosity * self.lap(self.ux) - self.p.friction * self.ux - adv_x
        )
        next_uy = self.uy + self.p.dt * (
            force_y + self.p.viscosity * self.lap(self.uy) - self.p.friction * self.uy - adv_y
        )
        self.project_velocity(next_ux, next_uy)

        uxx = self.ddx(self.ux)
        uxy = 0.5 * (self.ddy(self.ux) + self.ddx(self.uy))
        omega = 0.5 * (self.ddx(self.uy) - self.ddy(self.ux))
        s2 = self.qxx * self.qxx + self.qxy * self.qxy
        bulk = 1.35 * (self.p.target_s * self.p.target_s - s2)
        hxx = self.p.elastic * self.lap(self.qxx) + bulk * self.qxx
        hxy = self.p.elastic * self.lap(self.qxy) + bulk * self.qxy
        nudge = self.p.noise * 0.015 * self.rng.standard_normal(self.qxx.shape)

        next_qxx = self.qxx + self.p.dt * (
            self.p.gamma * hxx - 2 * omega * self.qxy + self.p.alignment * uxx - self.advect(self.qxx)
        ) + nudge
        next_qxy = self.qxy + self.p.dt * (
            self.p.gamma * hxy + 2 * omega * self.qxx + self.p.alignment * uxy - self.advect(self.qxy)
        ) + 0.6 * nudge
        mag = np.hypot(next_qxx, next_qxy)
        scale = np.ones_like(mag)
        mask = mag > 1.15
        scale[mask] = 1.15 / mag[mask]
        self.qxx = next_qxx * scale
        self.qxy = next_qxy * scale
        self.energy = float(np.mean(0.5 * self.p.elastic * (hxx * hxx + hxy * hxy) + 0.5 * (self.ux**2 + self.uy**2)))
        self.step_count += 1

    def count_defects(self) -> tuple[int, int]:
        theta = 0.5 * np.arctan2(self.qxy, self.qxx)
        a0 = theta
        a1 = np.roll(theta, -1, axis=1)
        a2 = np.roll(np.roll(theta, -1, axis=1), -1, axis=0)
        a3 = np.roll(theta, -1, axis=0)
        winding = wrap_half_turn(a1 - a0) + wrap_half_turn(a2 - a1) + wrap_half_turn(a3 - a2) + wrap_half_turn(a0 - a3)
        charge = winding / (2 * np.pi)
        positive = int(np.count_nonzero(charge > 0.22))
        negative = int(np.count_nonzero(charge < -0.22))
        return positive, negative


def wrap_half_turn(a: np.ndarray) -> np.ndarray:
    return (a + np.pi / 2) % np.pi - np.pi / 2


def save_snapshot(sim: ActiveNematic, output_dir: Path) -> None:
    np.savez_compressed(
        output_dir / "final_state.npz",
        qxx=sim.qxx,
        qxy=sim.qxy,
        ux=sim.ux,
        uy=sim.uy,
        pressure=sim.pressure,
        step=sim.step_count,
        energy=sim.energy,
    )


def save_png(sim: ActiveNematic, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PNG output")
        return

    theta = 0.5 * np.arctan2(sim.qxy, sim.qxx)
    order = np.hypot(sim.qxx, sim.qxy)
    vort = 0.5 * (sim.ddx(sim.uy) - sim.ddy(sim.ux))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(vort, cmap="coolwarm", origin="lower")
    axes[0].set_title("vorticity")
    axes[1].imshow(order, cmap="viridis", origin="lower")
    axes[1].set_title("|Q|")
    axes[2].imshow(theta, cmap="twilight", origin="lower")
    axes[2].set_title("director angle")
    for ax in axes:
        ax.set_axis_off()
    fig.savefig(output_dir / "final_snapshot.png", dpi=160)
    plt.close(fig)


def parse_args() -> Params:
    parser = argparse.ArgumentParser(description="Run the active nematic hydrodynamics reference simulator.")
    parser.add_argument("--grid", type=int, default=128, help="periodic grid cells per side")
    parser.add_argument("--steps", type=int, default=1000, help="number of time steps")
    parser.add_argument("--dt", type=float, default=0.18, help="time step")
    parser.add_argument("--activity", type=float, default=0.70, help="activity magnitude")
    parser.add_argument("--contractile", action="store_true", help="use contractile activity instead of extensile")
    parser.add_argument("--inactive", action="store_true", help="turn off active forcing")
    parser.add_argument("--elastic", type=float, default=0.23, help="elasticity K")
    parser.add_argument("--viscosity", type=float, default=0.24, help="viscosity eta")
    parser.add_argument("--friction", type=float, default=0.055, help="friction alpha")
    parser.add_argument("--alignment", type=float, default=0.62, help="flow-alignment lambda")
    parser.add_argument("--noise", type=float, default=0.09, help="Q-tensor noise amplitude")
    parser.add_argument("--sample-every", type=int, default=10, help="history sampling interval")
    parser.add_argument("--seed", type=int, default=4, help="random seed")
    parser.add_argument("--no-nonlinear-momentum", action="store_true", help="disable u dot grad u in momentum")
    parser.add_argument("--pressure-iterations", type=int, default=22, help="Jacobi iterations for pressure projection")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/active_nematic"), help="output directory")
    parser.add_argument("--save-png", action="store_true", help="save a summary PNG if matplotlib is installed")
    args = parser.parse_args()
    return Params(
        grid=args.grid,
        steps=args.steps,
        dt=args.dt,
        activity=args.activity,
        activity_sign=-1.0 if args.contractile else 1.0,
        active=not args.inactive,
        elastic=args.elastic,
        viscosity=args.viscosity,
        friction=args.friction,
        alignment=args.alignment,
        noise=args.noise,
        nonlinear_momentum=not args.no_nonlinear_momentum,
        pressure_iterations=args.pressure_iterations,
        sample_every=max(1, args.sample_every),
        seed=args.seed,
        output_dir=args.output_dir,
        save_png=args.save_png,
    )


def main() -> None:
    params = parse_args()
    params.output_dir.mkdir(parents=True, exist_ok=True)
    sim = ActiveNematic(params)
    history_path = params.output_dir / "history.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "energy", "positive_defects", "negative_defects"])
        for _ in range(params.steps):
            sim.step()
            if sim.step_count % params.sample_every == 0 or sim.step_count == params.steps:
                positive, negative = sim.count_defects()
                writer.writerow([sim.step_count, f"{sim.energy:.8g}", positive, negative])
                print(
                    f"step={sim.step_count:6d} energy={sim.energy:.5f} "
                    f"defects +/{positive} -/{negative}"
                )
    save_snapshot(sim, params.output_dir)
    if params.save_png:
        save_png(sim, params.output_dir)
    print(f"Wrote {history_path} and {params.output_dir / 'final_state.npz'}")


if __name__ == "__main__":
    main()
