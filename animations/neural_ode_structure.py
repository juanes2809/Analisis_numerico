"""
Neural ODE — vista por bloques + zoom al interior de cada uno.

Narrativa:
  1) Tres bloques: Datos → ODE solver (caja negra) → Campo vectorial fθ.
  2) Zoom sucesivo: qué hay dentro de cada bloque.
  3) En el último, el campo vectorial evoluciona (parámetro / tiempo).

Requisitos: pip install manim

GIF:
  manim -qm animations/neural_ode_structure.py NeuralODEStructure --format gif
"""

from __future__ import annotations

import numpy as np

from manim import *


def _norm2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v[:2])) + 1e-9
    return v / n


def vector_field_mobject(phase: float, x0: float, y0: float, scale: float = 0.42) -> VGroup:
    """Flechas ligeras (pocas) para poder animar fase sin matar el render."""
    g = VGroup()
    for xi in range(-4, 5):
        for yi in range(-3, 4):
            x = x0 + xi * scale * 0.62
            y = y0 + yi * scale * 0.58
            p = np.array([x, y, 0.0])
            # Campo 2D que rota y deforma con phase (sugiere θ o t en fθ)
            c, s = np.cos(phase), np.sin(phase)
            vx = c * (-0.55 * y) + s * (0.45 * x - 0.35 * y)
            vy = s * (0.55 * y) + c * (0.35 * x + 0.45 * y)
            v = np.array([vx, vy, 0.0])
            v = _norm2(v) * scale * 0.55
            hue = (phase * 0.4 + 0.15 * x + 0.12 * y) % 1.0
            col = interpolate_color(BLUE_E, RED_C, 0.5 + 0.5 * np.sin(TAU * hue))
            g.add(
                Arrow(
                    p,
                    p + v,
                    buff=0,
                    stroke_width=2.2,
                    max_tip_length_to_length_ratio=0.22,
                    color=col,
                )
            )
    return g


def mini_mlp() -> VGroup:
    rects = VGroup(
        *[Rectangle(width=0.14, height=0.45, stroke_width=1.5, color=BLUE_B).shift(RIGHT * i * 0.22) for i in range(4)]
    )
    return rects.scale(0.85)


class NeuralODEStructure(MovingCameraScene):
    def construct(self) -> None:
        title = Text("Neural ODE: datos → solver → campo vectorial", font_size=32, weight=BOLD)
        title.to_edge(UP, buff=0.2)

        # --- Cascos de los tres bloques (vista general) ---
        w, h = 2.85, 1.55
        r0 = 0.12

        shell_data = RoundedRectangle(corner_radius=r0, width=w, height=h, color=BLUE_C, stroke_width=3)
        lab_data = Text("Datos", font_size=28, weight=BOLD).next_to(shell_data, UP, buff=0.12)
        sub_data = Text("h(0), tiempo t,\nobservaciones…", font_size=20, color=GRAY_A, line_spacing=0.95)
        sub_data.move_to(shell_data.get_center())
        block_data = VGroup(shell_data, lab_data, sub_data).move_to(LEFT * 3.55)

        shell_box = RoundedRectangle(corner_radius=r0, width=w, height=h, color=GRAY_B, stroke_width=3)
        shell_box.set_fill(BLACK, opacity=0.92)
        lab_box = Text("ODE solver", font_size=26, weight=BOLD, color=WHITE).next_to(shell_box, UP, buff=0.12)
        sub_box = Text("caja negra", font_size=22, color=GRAY_B).move_to(shell_box.get_center())
        block_solver = VGroup(shell_box, lab_box, sub_box).move_to(ORIGIN)

        shell_vf = RoundedRectangle(corner_radius=r0, width=w, height=h, color=GREEN_C, stroke_width=3)
        lab_vf = Text("Campo vectorial", font_size=26, weight=BOLD).next_to(shell_vf, UP, buff=0.12)
        sub_vf = Text("dh/dt = fθ(h, t)", font_size=22, color=GRAY_A).move_to(shell_vf.get_center())
        block_vf = VGroup(shell_vf, lab_vf, sub_vf).move_to(RIGHT * 3.55)

        arr1 = Arrow(block_data.get_right(), block_solver.get_left(), buff=0.12, stroke_width=4, color=YELLOW)
        arr2 = Arrow(block_solver.get_right(), block_vf.get_left(), buff=0.12, stroke_width=4, color=YELLOW)

        overview = VGroup(title, block_data, block_solver, block_vf, arr1, arr2)

        # Interiores (detalle bajo zoom); empiezan invisibles
        # --- Interior datos ---
        plane_d = NumberPlane(
            x_range=(-1.2, 1.4, 1),
            y_range=(-0.8, 1.0, 1),
            x_length=2.2,
            y_length=1.5,
            background_line_style={"stroke_opacity": 0.35},
        ).move_to(shell_data.get_center()).scale(0.72)
        d0 = Dot(plane_d.c2p(0, 0.35), color=YELLOW, radius=0.09)
        t_axis = Arrow(plane_d.get_corner(DL) + RIGHT * 0.15 + UP * 0.1, plane_d.get_corner(DR) + LEFT * 0.15 + UP * 0.1, color=GRAY_A, stroke_width=2)
        cap_t = Text("t", font_size=20, color=GRAY_A).next_to(t_axis, DOWN, buff=0.06)
        cap_h = Text("h", font_size=20, color=GRAY_A).next_to(plane_d.y_axis, LEFT, buff=0.06)
        interior_data = VGroup(
            Text("Dentro: estado y tiempo", font_size=22, color=BLUE_A).next_to(plane_d, UP, buff=0.15),
            plane_d,
            d0,
            t_axis,
            cap_t,
            cap_h,
        )
        interior_data.set_opacity(0)

        # --- Interior caja negra ---
        inner_bg = RoundedRectangle(corner_radius=0.08, width=w * 0.92, height=h * 0.88, color=GRAY_E, stroke_width=1)
        inner_bg.set_fill("#1a1a1a", opacity=1).move_to(shell_box.get_center())
        steps = VGroup(
            *[
                DashedLine(
                    inner_bg.get_top() + DOWN * (0.35 + 0.22 * i) + LEFT * 0.9,
                    inner_bg.get_top() + DOWN * (0.35 + 0.22 * i) + RIGHT * 0.9,
                    color=GRAY_B,
                    stroke_width=1.5,
                )
                for i in range(5)
            ]
        )
        calls = Text("Muchas consultas a fθ\n(pasos internos del integrador)", font_size=19, color=WHITE, line_spacing=1.05)
        calls.move_to(inner_bg.get_center() + UP * 0.28)
        row_mlps = VGroup(*[mini_mlp().shift(RIGHT * i * 0.55) for i in range(3)]).scale(0.55)
        row_mlps.next_to(calls, DOWN, buff=0.22)
        interior_solver = VGroup(inner_bg, steps, calls, row_mlps)
        interior_solver.set_opacity(0)

        # --- Interior campo: plano + campo que cambia ---
        phase_tr = ValueTracker(0.0)
        cx, cy = shell_vf.get_center()[0], shell_vf.get_center()[1]
        plane_vf = NumberPlane(
            x_range=(-2.2, 2.2, 1),
            y_range=(-1.6, 1.6, 1),
            x_length=2.35,
            y_length=1.75,
            background_line_style={"stroke_opacity": 0.28},
        ).move_to(np.array([cx, cy, 0]))
        plane_vf.scale(0.78)
        cap_vf = Text("Dentro: flechas = dirección de dh/dt\n(cambia con θ y con t)", font_size=20, color=GREEN_A, line_spacing=1.05)
        cap_vf.next_to(plane_vf, UP, buff=0.12)
        plane_vf.set_opacity(0)
        cap_vf.set_opacity(0)

        def _vf_refresh() -> VGroup:
            c = plane_vf.get_center()
            return vector_field_mobject(phase_tr.get_value(), c[0], c[1], scale=0.38)

        vf_mob = always_redraw(_vf_refresh)

        self.add(overview)
        self.play(FadeIn(title), LaggedStart(FadeIn(block_data), FadeIn(block_solver), FadeIn(block_vf), lag_ratio=0.2))
        self.play(GrowArrow(arr1), GrowArrow(arr2), run_time=0.8)
        self.wait(0.25)

        # ========== Zoom 1: Datos ==========
        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale(0.48).move_to(block_data.get_center() + DOWN * 0.05),
            sub_data.animate.set_opacity(0.12),
            run_time=1.15,
            rate_func=smooth,
        )
        self.play(FadeIn(interior_data, shift=DOWN * 0.08), run_time=0.9)
        self.wait(0.35)
        self.play(FadeOut(interior_data), run_time=0.45)
        self.play(Restore(self.camera.frame), sub_data.animate.set_opacity(1), run_time=1.1, rate_func=smooth)

        # ========== Zoom 2: Solver ==========
        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(block_solver.get_center() + DOWN * 0.05),
            sub_box.animate.set_opacity(0.1),
            run_time=1.1,
            rate_func=smooth,
        )
        self.play(FadeIn(interior_solver, shift=UP * 0.1), run_time=0.85)
        self.play(LaggedStart(*[ShowPassingFlash(s, time_width=0.35) for s in steps], lag_ratio=0.12), run_time=1.2)
        self.wait(0.3)
        self.play(FadeOut(interior_solver), run_time=0.45)
        self.play(Restore(self.camera.frame), sub_box.animate.set_opacity(1), run_time=1.05, rate_func=smooth)

        # ========== Zoom 3: Campo vectorial ==========
        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale(0.52).move_to(block_vf.get_center() + DOWN * 0.08),
            sub_vf.animate.set_opacity(0.12),
            run_time=1.1,
            rate_func=smooth,
        )
        self.play(FadeIn(plane_vf), FadeIn(cap_vf), run_time=0.6)
        self.add(vf_mob)
        self.play(phase_tr.animate.set_value(2.7 * PI), run_time=3.2, rate_func=linear)
        self.wait(0.2)
        self.remove(vf_mob)
        self.play(FadeOut(plane_vf), FadeOut(cap_vf), run_time=0.45)
        self.play(Restore(self.camera.frame), sub_vf.animate.set_opacity(1), run_time=1.05, rate_func=smooth)

        cierre = Text(
            "Flujo: datos alimentan la caja negra; ésta integra el campo fθ",
            font_size=24,
        ).to_edge(DOWN, buff=0.18)
        self.play(FadeIn(cierre, shift=UP * 0.1))
        self.wait(0.8)
        self.play(FadeOut(cierre), FadeOut(overview), run_time=0.9)
