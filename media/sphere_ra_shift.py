"""
Manim scene: why a fixed 2 deg boresight shift spans more RA degrees near the pole.

Renders a 3B1B-style celestial sphere with an RA/DEC graticule, a 10x10 deg field at
DEC 30, a real boresight (blue) and a camera boresight (amber) offset by a constant 2 deg
true angle. The field then moves up to DEC 70 keeping the same 2 deg tilt, and the camera
travels with it (move_camera). The RA gap is read off (2.31 deg -> 5.85 deg = 2 / cos(DEC)).

Render:
    manim -pqh media/sphere_ra_shift.py RAShift          # 1080p
    manim -pql media/sphere_ra_shift.py RAShift           # fast preview
"""
from __future__ import annotations

import numpy as np
import imageio_ffmpeg
from manim import config
config.ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()  # use the pip-bundled ffmpeg

from manim import (
    ThreeDScene, Sphere, Surface, VGroup, Dot3D, Line3D, Text, RoundedRectangle,
    Create, FadeIn, FadeOut, Transform, ParametricFunction,
    BLUE, YELLOW, TEAL, GREY_B, GREY_D, BLACK, WHITE, DEGREES, UP, DOWN, RIGHT,
)

R = 2.0  # sphere radius (scene units)


def sph(ra_deg, dec_deg, r=R):
    """(RA, DEC) in degrees -> 3D point on a sphere of radius r."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([r * np.cos(dec) * np.cos(ra),
                     r * np.cos(dec) * np.sin(ra),
                     r * np.sin(dec)])


def graticule():
    """Latitude circles + longitude meridians as thin faint curves."""
    lines = VGroup()
    for dec in range(-60, 90, 15):  # latitude circles
        lines.add(ParametricFunction(lambda t, d=dec: sph(np.degrees(t), d),
                                     t_range=[-np.pi, np.pi, 0.05],
                                     color=GREY_D, stroke_width=1.2))
    for ra in range(0, 360, 10):    # meridians every 10 deg (converge at the pole)
        lines.add(ParametricFunction(lambda t, a=ra: sph(a, np.degrees(t)),
                                     t_range=[-np.pi / 2, np.pi / 2, 0.05],
                                     color=GREY_D, stroke_width=1.2))
    return lines


def field_box(dec_c, color=TEAL):
    """A 10x10 deg field centred at (RA=0, DEC=dec_c): half-RA grows as 5/cos(dec)."""
    half_ra = 5.0 / np.cos(np.radians(dec_c))
    corners = [(-half_ra, dec_c + 5), (half_ra, dec_c + 5),
               (half_ra, dec_c - 5), (-half_ra, dec_c - 5)]
    edges = VGroup()
    for (ra0, d0), (ra1, d1) in zip(corners, corners[1:] + corners[:1]):
        edges.add(ParametricFunction(
            lambda t, a0=ra0, a1=ra1, b0=d0, b1=d1: sph(a0 + (a1 - a0) * t, b0 + (b1 - b0) * t),
            t_range=[0, 1, 0.02], color=color, stroke_width=4))
    return edges


class RAShift(ThreeDScene):
    def construct(self):
        # Look straight down the RA=0 meridian (theta=0 -> +x hemisphere faces us),
        # tilted up so DEC 30..70 sits in the upper-centre of the disc.
        self.set_camera_orientation(phi=70 * DEGREES, theta=0 * DEGREES, zoom=1.15,
                                    frame_center=[0, 0, 0.7])

        globe = Sphere(radius=R, resolution=(32, 32), fill_opacity=0.12,
                       stroke_width=0, checkerboard_colors=[GREY_D, GREY_D])
        grid = graticule()
        pole = Dot3D(sph(0, 90), color=GREY_B, radius=0.05)
        self.play(Create(globe), run_time=1.0)
        self.play(Create(grid), run_time=2.0)
        self.add(pole)

        # --- 2D title/read-out overlay (fixed to the screen, not the sphere) ---
        title = Text("A fixed 2° boresight shift = more RA degrees near the pole",
                     font_size=26, color=WHITE)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(title))

        # --- field + boresights at DEC 30 ---
        dec_c = 30
        box = field_box(dec_c)
        real = Dot3D(sph(0, dec_c), color=BLUE, radius=0.02)
        shift_ra = 2.0 / np.cos(np.radians(dec_c))
        cam = Dot3D(sph(shift_ra, dec_c), color=YELLOW, radius=0.02)
        tilt = Line3D(sph(0, dec_c), sph(shift_ra, dec_c), color=YELLOW, thickness=0.04)
        self.play(Create(box), FadeIn(real))
        self.play(Create(tilt), FadeIn(cam))

        readout = self._readout(30, shift_ra)
        self.add_fixed_in_frame_mobjects(readout)
        self.play(FadeIn(readout))
        self.wait(1.5)

        # --- move the field up to DEC 70, keep the same 2 deg true tilt ---
        dec_c2 = 70
        box2 = field_box(dec_c2)
        shift_ra2 = 2.0 / np.cos(np.radians(dec_c2))
        real2 = Dot3D(sph(0, dec_c2), color=BLUE, radius=0.02)
        cam2 = Dot3D(sph(shift_ra2, dec_c2), color=YELLOW, radius=0.02)
        tilt2 = Line3D(sph(0, dec_c2), sph(shift_ra2, dec_c2), color=YELLOW, thickness=0.02)
        readout2 = self._readout(70, shift_ra2)
        self.add_fixed_in_frame_mobjects(readout2)
        readout2.set_opacity(0)

        # Move the camera WITH the field: move_camera animates the camera, and added_anims
        # play at the same time. Tilt further down (phi) and lift the look-at point
        # (frame_center z) so DEC 70 stays centred as the field climbs toward the pole.
        self.move_camera(
            phi=52 * DEGREES, theta=0 * DEGREES, zoom=1.35, frame_center=[0, 0, 1.5],
            added_anims=[
                Transform(box, box2), Transform(real, real2),
                Transform(cam, cam2), Transform(tilt, tilt2),
                Transform(readout, readout2.copy().set_opacity(1)),
            ],
            run_time=3.0,
        )
        self.wait(2.0)

    def _readout(self, dec_c, shift_ra):
        """Fixed-frame text block: DEC, the RA gap, and the 2/cos(DEC) identity."""
        lines = VGroup(
            Text(f"Field at DEC {dec_c}°", font_size=24, color=WHITE),
            Text("real  RA = 0.00°", font_size=20, color=BLUE),
            Text(f"camera RA = +{shift_ra:.2f}°", font_size=20, color=YELLOW),
            Text(f"RA gap = 2° / cos({dec_c}°) = {shift_ra:.2f}°", font_size=20, color=YELLOW),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.18)
        panel = RoundedRectangle(corner_radius=0.12, width=lines.width + 0.5,
                                 height=lines.height + 0.4, fill_color=BLACK,
                                 fill_opacity=0.72, stroke_color=GREY_D, stroke_width=1.5)
        panel.move_to(lines)
        group = VGroup(panel, lines)
        group.to_corner(RIGHT + UP).shift([0, -1.4, 0])
        return group
