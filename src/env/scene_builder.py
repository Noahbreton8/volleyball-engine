"""
Programmatically generates the MuJoCo XML scene with all 12 players + court.
Each player body is named p{i}_* where i is the global player index (0-11).
"""
import numpy as np
from .constants import (
    TEAM_A_START_POSITIONS, TEAM_B_START_POSITIONS,
    START_HEIGHT, SIM_DT, PLAYERS_PER_TEAM
)


def _player_body(pid: int, x: float, y: float) -> str:
    """Generate MJCF XML for one player body at position (x, y, START_HEIGHT)."""
    p = f"p{pid}"
    z = START_HEIGHT

    return f"""
    <body name="{p}_torso" pos="{x:.3f} {y:.3f} {z:.3f}">
      <freejoint name="{p}_root"/>
      <geom name="{p}_torso_g" type="capsule" fromto="0 0 -0.25 0 0 0.25"
            size="0.12" mass="25" rgba="{'0.4 0.6 0.9 1' if pid < 6 else '0.9 0.4 0.4 1'}"/>

      <body name="{p}_head" pos="0 0 0.38">
        <geom name="{p}_head_g" type="sphere" size="0.12" mass="5"/>
      </body>

      <!-- Left arm -->
      <body name="{p}_l_uarm" pos="-0.18 0 0.2">
        <joint name="{p}_lsho_x" axis="1 0 0" range="-3.1 1.5"  damping="2.0" stiffness="5" armature="0.02"/>
        <joint name="{p}_lsho_y" axis="0 1 0" range="-1.5 3.1"  damping="2.0" stiffness="5" armature="0.02"/>
        <joint name="{p}_lsho_z" axis="0 0 1" range="-1.5 1.5"  damping="2.0" stiffness="5" armature="0.02"/>
        <geom name="{p}_l_uarm_g" type="capsule" fromto="0 0 0 0 0 -0.28" size="0.06" mass="2.5"/>
        <body name="{p}_l_farm" pos="0 0 -0.28">
          <joint name="{p}_lelbow" axis="0 1 0" range="-2.5 0" damping="1.5" stiffness="3" armature="0.01"/>
          <geom name="{p}_l_farm_g" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.05" mass="1.5"/>
          <body name="{p}_l_hand" pos="0 0 -0.25">
            <joint name="{p}_lwrist" axis="0 1 0" range="-1.0 1.0" damping="1.0" stiffness="2" armature="0.005"/>
            <geom name="{p}_l_hand_g" type="sphere" size="0.05" mass="0.5"/>
            <site name="{p}_l_hand_site" pos="0 0 -0.05" size="0.01"/>
          </body>
        </body>
      </body>

      <!-- Right arm -->
      <body name="{p}_r_uarm" pos="0.18 0 0.2">
        <joint name="{p}_rsho_x" axis="1 0 0" range="-3.1 1.5"  damping="2.0" stiffness="5" armature="0.02"/>
        <joint name="{p}_rsho_y" axis="0 1 0" range="-3.1 1.5"  damping="2.0" stiffness="5" armature="0.02"/>
        <joint name="{p}_rsho_z" axis="0 0 1" range="-1.5 1.5"  damping="2.0" stiffness="5" armature="0.02"/>
        <geom name="{p}_r_uarm_g" type="capsule" fromto="0 0 0 0 0 -0.28" size="0.06" mass="2.5"/>
        <body name="{p}_r_farm" pos="0 0 -0.28">
          <joint name="{p}_relbow" axis="0 1 0" range="-2.5 0" damping="1.5" stiffness="3" armature="0.01"/>
          <geom name="{p}_r_farm_g" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.05" mass="1.5"/>
          <body name="{p}_r_hand" pos="0 0 -0.25">
            <joint name="{p}_rwrist" axis="0 1 0" range="-1.0 1.0" damping="1.0" stiffness="2" armature="0.005"/>
            <geom name="{p}_r_hand_g" type="sphere" size="0.05" mass="0.5"/>
            <site name="{p}_r_hand_site" pos="0 0 -0.05" size="0.01"/>
          </body>
        </body>
      </body>

      <!-- Pelvis + legs -->
      <body name="{p}_pelvis" pos="0 0 -0.3">
        <joint name="{p}_lumb_x" axis="1 0 0" range="-0.8 0.8" damping="3.0" stiffness="8" armature="0.05"/>
        <joint name="{p}_lumb_y" axis="0 1 0" range="-0.5 0.5" damping="3.0" stiffness="8" armature="0.05"/>
        <joint name="{p}_lumb_z" axis="0 0 1" range="-0.8 0.8" damping="3.0" stiffness="8" armature="0.05"/>
        <geom name="{p}_pelvis_g" type="capsule" fromto="0 0 -0.15 0 0 0.15" size="0.10" mass="10"/>

        <!-- Left leg -->
        <body name="{p}_l_thigh" pos="-0.1 0 -0.15">
          <joint name="{p}_lhip_x" axis="1 0 0" range="-0.5 1.5" damping="4.0" stiffness="10" armature="0.1"/>
          <joint name="{p}_lhip_y" axis="0 1 0" range="-0.5 0.5" damping="4.0" stiffness="10" armature="0.1"/>
          <joint name="{p}_lhip_z" axis="0 0 1" range="-1.5 0.8" damping="4.0" stiffness="10" armature="0.1"/>
          <geom name="{p}_l_thigh_g" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.08" mass="8"/>
          <body name="{p}_l_shin" pos="0 0 -0.42">
            <joint name="{p}_lknee" axis="0 1 0" range="-2.5 0" damping="3.0" stiffness="8" armature="0.05"/>
            <geom name="{p}_l_shin_g" type="capsule" fromto="0 0 0 0 0 -0.38" size="0.065" mass="4"/>
            <body name="{p}_l_foot" pos="0 0 -0.38">
              <joint name="{p}_lank_x" axis="1 0 0" range="-0.6 0.8" damping="2.0" stiffness="5" armature="0.02"/>
              <joint name="{p}_lank_y" axis="0 1 0" range="-0.4 0.4" damping="2.0" stiffness="5" armature="0.02"/>
              <geom name="{p}_l_foot_g" type="capsule" fromto="-0.05 0 0 0.18 0 0"
                    size="0.055" mass="1.5" friction="1.0 0.1 0.1"/>
            </body>
          </body>
        </body>

        <!-- Right leg -->
        <body name="{p}_r_thigh" pos="0.1 0 -0.15">
          <joint name="{p}_rhip_x" axis="1 0 0" range="-0.5 1.5" damping="4.0" stiffness="10" armature="0.1"/>
          <joint name="{p}_rhip_y" axis="0 1 0" range="-0.5 0.5" damping="4.0" stiffness="10" armature="0.1"/>
          <joint name="{p}_rhip_z" axis="0 0 1" range="-0.8 1.5" damping="4.0" stiffness="10" armature="0.1"/>
          <geom name="{p}_r_thigh_g" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.08" mass="8"/>
          <body name="{p}_r_shin" pos="0 0 -0.42">
            <joint name="{p}_rknee" axis="0 1 0" range="-2.5 0" damping="3.0" stiffness="8" armature="0.05"/>
            <geom name="{p}_r_shin_g" type="capsule" fromto="0 0 0 0 0 -0.38" size="0.065" mass="4"/>
            <body name="{p}_r_foot" pos="0 0 -0.38">
              <joint name="{p}_rank_x" axis="1 0 0" range="-0.6 0.8" damping="2.0" stiffness="5" armature="0.02"/>
              <joint name="{p}_rank_y" axis="0 1 0" range="-0.4 0.4" damping="2.0" stiffness="5" armature="0.02"/>
              <geom name="{p}_r_foot_g" type="capsule" fromto="-0.05 0 0 0.18 0 0"
                    size="0.055" mass="1.5" friction="1.0 0.1 0.1"/>
            </body>
          </body>
        </body>

      </body><!-- pelvis -->
    </body><!-- torso -->"""


def _player_actuators(pid: int) -> str:
    p = f"p{pid}"
    joints = [
        f"{p}_lsho_x", f"{p}_lsho_y", f"{p}_lsho_z",
        f"{p}_lelbow", f"{p}_lwrist",
        f"{p}_rsho_x", f"{p}_rsho_y", f"{p}_rsho_z",
        f"{p}_relbow", f"{p}_rwrist",
        f"{p}_lumb_x", f"{p}_lumb_y", f"{p}_lumb_z",
        f"{p}_lhip_x", f"{p}_lhip_y", f"{p}_lhip_z",
        f"{p}_lknee", f"{p}_lank_x", f"{p}_lank_y",
        f"{p}_rhip_x", f"{p}_rhip_y", f"{p}_rhip_z",
        f"{p}_rknee", f"{p}_rank_x", f"{p}_rank_y",
    ]
    lines = []
    for j in joints:
        gear = 150 if any(s in j for s in ["hip", "knee", "lumb"]) else 80
        lines.append(
            f'  <motor name="act_{j}" joint="{j}" gear="{gear}" '
            f'ctrllimited="true" ctrlrange="-1 1"/>'
        )
    return "\n".join(lines)


def build_scene_xml() -> str:
    """Return the complete MuJoCo XML string for the volleyball scene."""
    all_positions = (
        [(x, y) for x, y in TEAM_A_START_POSITIONS] +
        [(x, y) for x, y in TEAM_B_START_POSITIONS]
    )

    player_bodies = "\n".join(
        _player_body(i, x, y) for i, (x, y) in enumerate(all_positions)
    )

    player_actuators = "\n".join(
        _player_actuators(i) for i in range(len(all_positions))
    )

    # Hand-contact sensor sites for detecting ball-player contact
    hand_sites = "\n".join(
        f'  <touch name="touch_p{i}_l" site="p{i}_l_hand_site"/>\n'
        f'  <touch name="touch_p{i}_r" site="p{i}_r_hand_site"/>'
        for i in range(len(all_positions))
    )

    return f"""<mujoco model="volleyball">
  <compiler angle="radian" autolimits="true" meshdir="." texturedir="."/>

  <option timestep="{SIM_DT:.6f}" gravity="0 0 -9.81"
          iterations="50" solver="Newton" tolerance="1e-10">
    <flag contact="enable"/>
  </option>

  <size nconmax="500" njmax="2000"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048"/>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <material name="court_floor"    rgba="0.82 0.70 0.45 1"/>
    <material name="court_line"     rgba="1 1 1 1"/>
    <material name="net_mat"        rgba="0.25 0.25 0.25 0.85"/>
    <material name="ball_mat"       rgba="0.95 0.50 0.10 1"/>
    <material name="antenna_mat"    rgba="0.9 0.1 0.1 1"/>
  </asset>

  <worldbody>
    <!-- Court floor -->
    <geom name="floor" type="plane" pos="0 0 0" size="15 8 0.1"
          material="court_floor" condim="6" friction="0.65 0.1 0.1"/>

    <!-- Side lines -->
    <geom name="sl1" type="box" pos="9.0 0 0.001"  size="0.04 4.55 0.001" material="court_line" contype="0" conaffinity="0"/>
    <geom name="sl2" type="box" pos="-9.0 0 0.001" size="0.04 4.55 0.001" material="court_line" contype="0" conaffinity="0"/>
    <!-- End lines -->
    <geom name="el1" type="box" pos="0 4.5 0.001"  size="9.05 0.04 0.001" material="court_line" contype="0" conaffinity="0"/>
    <geom name="el2" type="box" pos="0 -4.5 0.001" size="9.05 0.04 0.001" material="court_line" contype="0" conaffinity="0"/>
    <!-- Attack lines -->
    <geom name="al1" type="box" pos="3.0 0 0.001"  size="0.04 4.5 0.001"  material="court_line" contype="0" conaffinity="0"/>
    <geom name="al2" type="box" pos="-3.0 0 0.001" size="0.04 4.5 0.001"  material="court_line" contype="0" conaffinity="0"/>

    <!-- Net -->
    <geom name="net" type="box" pos="0 0 1.215" size="0.04 4.5 1.215"
          material="net_mat" condim="3" friction="0.3 0.1 0.1"/>
    <geom name="net_post_1" type="cylinder" pos="0 4.65 1.25" size="0.04 1.26" material="net_mat"/>
    <geom name="net_post_2" type="cylinder" pos="0 -4.65 1.25" size="0.04 1.26" material="net_mat"/>
    <geom name="antenna_1" type="cylinder" pos="0 4.5 2.63" size="0.008 0.42" material="antenna_mat" contype="0" conaffinity="0"/>
    <geom name="antenna_2" type="cylinder" pos="0 -4.5 2.63" size="0.008 0.42" material="antenna_mat" contype="0" conaffinity="0"/>

    <!-- Volleyball (freejoint so physics applies) -->
    <body name="ball" pos="7.0 0.0 1.8">
      <freejoint name="ball_free"/>
      <geom name="ball_g" type="sphere" size="{0.103}" mass="{0.270}"
            material="ball_mat" condim="6"
            friction="0.35 0.05 0.05" solimp="0.99 0.9999 0.001 0.5 2" solref="0.001 0.3"/>
    </body>

    <!-- Player bodies -->
{player_bodies}

  </worldbody>

  <actuator>
{player_actuators}
  </actuator>

  <sensor>
    <framepos name="ball_pos"  objtype="body" objname="ball"/>
    <framelinvel name="ball_velp" objtype="body" objname="ball"/>
    <frameangvel name="ball_spin" objtype="body" objname="ball"/>
{hand_sites}
  </sensor>

</mujoco>"""
