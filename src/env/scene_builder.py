"""
Generates MuJoCo XML scene with 12 standard humanoid players + volleyball court.
Each player body is named p{i}_* where i is the global player index (0-11).
Body structure mirrors the standard MuJoCo Gymnasium Humanoid-v4 model.
"""
from .constants import (
    TEAM_A_START_POSITIONS, TEAM_B_START_POSITIONS,
    START_HEIGHT, SIM_DT, PLAYERS_PER_TEAM
)


def _player_body(pid: int, x: float, y: float) -> str:
    """One prefixed copy of the standard MuJoCo humanoid body at (x, y, START_HEIGHT)."""
    P = f"p{pid}_"
    color = "0.4 0.6 0.9 1" if pid < PLAYERS_PER_TEAM else "0.9 0.4 0.4 1"
    z = START_HEIGHT

    return f"""    <body name="{P}torso" pos="{x:.3f} {y:.3f} {z:.3f}">
      <joint name="{P}root" type="free" damping="0" stiffness="0" armature="0" limited="false"/>
      <geom fromto="0 -.07 0 0 .07 0" name="{P}torso1" size="0.07" type="capsule" rgba="{color}"/>
      <geom name="{P}head" pos="0 0 .19" size=".09" type="sphere" rgba="{color}"/>
      <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="{P}uwaist" size="0.06" type="capsule" rgba="{color}"/>
      <body name="{P}lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
        <geom fromto="0 -.06 0 0 .06 0" name="{P}lwaist_g" size="0.06" type="capsule" rgba="{color}"/>
        <joint armature="0.02" axis="0 0 1" damping="5" name="{P}abdomen_z" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
        <joint armature="0.02" axis="0 1 0" damping="5" name="{P}abdomen_y" pos="0 0 0.065" range="-75 30" stiffness="10" type="hinge"/>
        <body name="{P}pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
          <joint armature="0.02" axis="1 0 0" damping="5" name="{P}abdomen_x" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
          <geom fromto="-.02 -.07 0 -.02 .07 0" name="{P}butt" size="0.09" type="capsule" rgba="{color}"/>
          <body name="{P}right_thigh" pos="0 -0.1 -0.04">
            <joint armature="0.01" axis="1 0 0" damping="5" name="{P}right_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
            <joint armature="0.01" axis="0 0 1" damping="5" name="{P}right_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
            <joint armature="0.0080" axis="0 1 0" damping="5" name="{P}right_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
            <geom fromto="0 0 0 0 0.01 -.34" name="{P}right_thigh1" size="0.06" type="capsule" rgba="{color}"/>
            <body name="{P}right_shin" pos="0 0.01 -0.403">
              <joint armature="0.0060" axis="0 -1 0" name="{P}right_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
              <geom fromto="0 0 0 0 0 -.3" name="{P}right_shin1" size="0.049" type="capsule" rgba="{color}"/>
              <body name="{P}right_foot" pos="0 0 -0.45">
                <geom name="{P}right_foot_g" pos="0 0 0.1" size="0.075" type="sphere" rgba="{color}" friction="1 0.1 0.1" condim="3"/>
              </body>
            </body>
          </body>
          <body name="{P}left_thigh" pos="0 0.1 -0.04">
            <joint armature="0.01" axis="-1 0 0" damping="5" name="{P}left_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
            <joint armature="0.01" axis="0 0 -1" damping="5" name="{P}left_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
            <joint armature="0.01" axis="0 1 0" damping="5" name="{P}left_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
            <geom fromto="0 0 0 0 -0.01 -.34" name="{P}left_thigh1" size="0.06" type="capsule" rgba="{color}"/>
            <body name="{P}left_shin" pos="0 -0.01 -0.403">
              <joint armature="0.0060" axis="0 -1 0" name="{P}left_knee" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
              <geom fromto="0 0 0 0 0 -.3" name="{P}left_shin1" size="0.049" type="capsule" rgba="{color}"/>
              <body name="{P}left_foot" pos="0 0 -0.45">
                <geom name="{P}left_foot_g" type="sphere" size="0.075" pos="0 0 0.1" rgba="{color}" friction="1 0.1 0.1" condim="3"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="{P}right_upper_arm" pos="0 -0.17 0.06">
        <joint armature="0.0068" axis="2 1 1" name="{P}right_shoulder1" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
        <joint armature="0.0051" axis="0 -1 1" name="{P}right_shoulder2" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
        <geom fromto="0 0 0 .16 -.16 -.16" name="{P}right_uarm1" size="0.04 0.16" type="capsule" rgba="{color}"/>
        <body name="{P}right_lower_arm" pos=".18 -.18 -.18">
          <joint armature="0.0028" axis="0 -1 1" name="{P}right_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
          <geom fromto="0.01 0.01 0.01 .17 .17 .17" name="{P}right_larm" size="0.031" type="capsule" rgba="{color}"/>
          <geom name="{P}right_hand" pos=".18 .18 .18" size="0.04" type="sphere" rgba="{color}"/>
          <site name="{P}right_hand_site" pos=".18 .18 .18" size="0.01"/>
        </body>
      </body>
      <body name="{P}left_upper_arm" pos="0 0.17 0.06">
        <joint armature="0.0068" axis="2 -1 1" name="{P}left_shoulder1" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
        <joint armature="0.0051" axis="0 1 1" name="{P}left_shoulder2" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
        <geom fromto="0 0 0 .16 .16 -.16" name="{P}left_uarm1" size="0.04 0.16" type="capsule" rgba="{color}"/>
        <body name="{P}left_lower_arm" pos=".18 .18 -.18">
          <joint armature="0.0028" axis="0 -1 -1" name="{P}left_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
          <geom fromto="0.01 -0.01 0.01 .17 -.17 .17" name="{P}left_larm" size="0.031" type="capsule" rgba="{color}"/>
          <geom name="{P}left_hand" pos=".18 -.18 .18" size="0.04" type="sphere" rgba="{color}"/>
          <site name="{P}left_hand_site" pos=".18 -.18 .18" size="0.01"/>
        </body>
      </body>
    </body>"""


def _player_tendons(pid: int) -> str:
    P = f"p{pid}_"
    return (
        f'    <fixed name="{P}left_hipknee">\n'
        f'      <joint coef="-1" joint="{P}left_hip_y"/>\n'
        f'      <joint coef="1" joint="{P}left_knee"/>\n'
        f'    </fixed>\n'
        f'    <fixed name="{P}right_hipknee">\n'
        f'      <joint coef="-1" joint="{P}right_hip_y"/>\n'
        f'      <joint coef="1" joint="{P}right_knee"/>\n'
        f'    </fixed>'
    )


def _player_actuators(pid: int) -> str:
    P = f"p{pid}_"
    # Gear values match standard MuJoCo humanoid (legs strong, arms weak)
    motors = [
        (f"{P}abdomen_y",    100),
        (f"{P}abdomen_z",    100),
        (f"{P}abdomen_x",    100),
        (f"{P}right_hip_x",  100),
        (f"{P}right_hip_z",  100),
        (f"{P}right_hip_y",  300),
        (f"{P}right_knee",   200),
        (f"{P}left_hip_x",   100),
        (f"{P}left_hip_z",   100),
        (f"{P}left_hip_y",   300),
        (f"{P}left_knee",    200),
        (f"{P}right_shoulder1", 25),
        (f"{P}right_shoulder2", 25),
        (f"{P}right_elbow",  25),
        (f"{P}left_shoulder1",  25),
        (f"{P}left_shoulder2",  25),
        (f"{P}left_elbow",   25),
    ]
    lines = [
        f'    <motor gear="{gear}" joint="{joint}" name="{joint}" ctrllimited="true" ctrlrange="-.4 .4"/>'
        for joint, gear in motors
    ]
    return "\n".join(lines)


def build_scene_xml() -> str:
    """Return the complete MuJoCo XML string for the volleyball scene."""
    all_positions = (
        list(TEAM_A_START_POSITIONS) +
        list(TEAM_B_START_POSITIONS)
    )

    player_bodies = "\n".join(
        _player_body(i, x, y) for i, (x, y) in enumerate(all_positions)
    )
    player_tendons = "\n".join(
        _player_tendons(i) for i in range(len(all_positions))
    )
    player_actuators = "\n".join(
        _player_actuators(i) for i in range(len(all_positions))
    )

    # Hand sites for reach reward observation
    hand_sites = "\n".join(
        f'    <framepos name="touch_p{i}_l" objtype="site" objname="p{i}_left_hand_site"/>\n'
        f'    <framepos name="touch_p{i}_r" objtype="site" objname="p{i}_right_hand_site"/>'
        for i in range(len(all_positions))
    )

    return f"""<mujoco model="volleyball">
  <compiler angle="degree" inertiafromgeom="true" autolimits="true"/>

  <option timestep="{SIM_DT:.6f}" gravity="0 0 -9.81"
          integrator="RK4" iterations="50" solver="Newton" tolerance="1e-10">
    <flag contact="enable"/>
  </option>

  <size nconmax="1000" njmax="4000"/>

  <default>
    <geom conaffinity="1" condim="1" contype="1" margin="0.001"/>
    <motor ctrllimited="true" ctrlrange="-.4 .4"/>
  </default>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048"/>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <material name="court_floor" rgba="0.82 0.70 0.45 1"/>
    <material name="court_line"  rgba="1 1 1 1"/>
    <material name="net_mat"     rgba="0.25 0.25 0.25 0.85"/>
    <material name="ball_mat"    rgba="0.95 0.50 0.10 1"/>
    <material name="antenna_mat" rgba="0.9 0.1 0.1 1"/>
  </asset>

  <worldbody>
    <!-- Court floor -->
    <geom name="floor" type="plane" pos="0 0 0" size="15 8 0.1"
          material="court_floor" condim="6" friction="0.65 0.1 0.1"/>

    <!-- Court lines (non-colliding) -->
    <geom name="sl1" type="box" pos="9.0 0 0.001"  size="0.04 4.55 0.001" material="court_line" contype="0" conaffinity="0"/>
    <geom name="sl2" type="box" pos="-9.0 0 0.001" size="0.04 4.55 0.001" material="court_line" contype="0" conaffinity="0"/>
    <geom name="el1" type="box" pos="0 4.5 0.001"  size="9.05 0.04 0.001" material="court_line" contype="0" conaffinity="0"/>
    <geom name="el2" type="box" pos="0 -4.5 0.001" size="9.05 0.04 0.001" material="court_line" contype="0" conaffinity="0"/>
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

    <!-- Player bodies (12 standard humanoids, prefixed p0_ .. p11_) -->
{player_bodies}

  </worldbody>

  <tendon>
{player_tendons}
  </tendon>

  <actuator>
{player_actuators}
  </actuator>

  <sensor>
    <framepos name="ball_pos"    objtype="body" objname="ball"/>
    <framelinvel name="ball_velp" objtype="body" objname="ball"/>
    <frameangvel name="ball_spin" objtype="body" objname="ball"/>
{hand_sites}
  </sensor>

</mujoco>"""
