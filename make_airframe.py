"""Generate a gym-pybullet-drones URDF for an arbitrary quadrotor.

Every policy trained in this project so far is specific to the Crazyflie 2.X in the vendored
assets: 27g, 12.6cm across, 2.25:1 thrust-to-weight. That is an indoor lab micro-drone, and
several limits we spent the day characterising are its limits rather than the controller's --
the ~2.5 m/s cruise ceiling, the thrust saturation that made tall climbs tip over, and a wind
sensitivity where a gust of 11% of the aircraft's weight was a serious disturbance.

A machine that carries lidar plus a camera plus compute is a different aircraft by two orders
of magnitude in mass, so rather than hard-coding another guess this builds the URDF from the
numbers that actually matter. Physical parameters that are awkward to measure are derived from
mass and geometry using standard approximations, documented per-field below.

    uv run python make_airframe.py --name heavy --mass 2.5 --wheelbase 0.45 --thrust2weight 2.2
"""
import argparse
import os

TEMPLATE = """<?xml version="1.0" ?>
<robot name="{name}">
  <properties arm="{arm:.4f}" kf="{kf:.6e}" km="{km:.6e}" thrust2weight="{t2w}" max_speed_kmh="{max_kmh}" gnd_eff_coeff="11.36859" prop_radius="{prop_radius:.6e}" drag_coeff_xy="{drag_xy:.6e}" drag_coeff_z="{drag_z:.6e}" dw_coeff_1="2267.18" dw_coeff_2=".16" dw_coeff_3="-.11" />
  <link name="base_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx:.6e}" ixy="0.0" ixz="0.0" iyy="{iyy:.6e}" iyz="0.0" izz="{izz:.6e}"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><cylinder radius="{body_radius:.4f}" length="{body_height:.4f}"/></geometry>
      <material name="grey"><color rgba=".5 .5 .5 1"/></material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><cylinder radius="{body_radius:.4f}" length="{body_height:.4f}"/></geometry>
    </collision>
  </link>
{props}
</robot>
"""

PROP = """  <link name="prop{i}_link">
    <inertial>
      <origin rpy="0 0 0" xyz="{x:.4f} {y:.4f} 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop{i}_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop{i}_link"/>
  </joint>
"""

CENTER = """  <link name="center_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="center_joint" type="fixed">
    <parent link="base_link"/>
    <child link="center_link"/>
  </joint>
"""


def build(name, mass, wheelbase, t2w, max_kmh, prop_radius=None, out_dir=None):
    arm = wheelbase / 2.0
    # Props scale with the airframe; a quad's props are typically just under half the arm
    # spacing so adjacent discs do not overlap.
    prop_radius = prop_radius if prop_radius else arm * 0.45

    # Inertia: model the airframe as a uniform disc of radius=arm for izz, and use the standard
    # thin-disc relation ixx = iyy = izz/2 for the in-plane axes. Crude next to a CAD model, but
    # right to within a factor that matters far less than mass and thrust, and it reproduces the
    # Crazyflie's published values to the correct order of magnitude.
    izz = 0.5 * mass * arm ** 2
    ixx = iyy = izz / 2.0

    # kf relates thrust to squared rotor speed: T = kf * rpm^2. Pin it so that all four rotors
    # at MAX_RPM give exactly thrust2weight * weight, which is the relation BaseAviary assumes
    # when it derives MAX_RPM back out of kf and thrust2weight.
    max_rpm = 25000.0 if mass < 0.5 else 12000.0  # bigger props turn slower
    kf = (t2w * mass * 9.81) / (4.0 * max_rpm ** 2)
    # Torque coefficient: typical km/kf for hobby propellers is ~0.02 * prop_radius.
    km = kf * 0.02 * prop_radius

    # Drag grows with frontal area; scale the Crazyflie's coefficients by area ratio.
    area_ratio = (arm / 0.0397) ** 2
    drag_xy = 9.1785e-7 * area_ratio
    drag_z = 10.311e-7 * area_ratio

    props = "".join(
        PROP.format(i=i, x=dx * arm * 0.7071, y=dy * arm * 0.7071)
        for i, (dx, dy) in enumerate([(1, -1), (1, 1), (-1, 1), (-1, -1)])
    ) + CENTER

    urdf = TEMPLATE.format(
        name=name, arm=arm, kf=kf, km=km, t2w=t2w, max_kmh=max_kmh,
        prop_radius=prop_radius, drag_xy=drag_xy, drag_z=drag_z, mass=mass,
        ixx=ixx, iyy=iyy, izz=izz,
        body_radius=arm * 0.6, body_height=max(0.05, arm * 0.3), props=props,
    )

    out_dir = out_dir or "gym-pybullet-drones/gym_pybullet_drones/assets"
    path = os.path.join(out_dir, f"{name}.urdf")
    with open(path, "w") as fh:
        fh.write(urdf)

    hover_rpm = (9.81 * mass / (4 * kf)) ** 0.5
    print(f"wrote {path}")
    print(f"  mass            {mass*1000:.0f} g")
    print(f"  wheelbase       {wheelbase*100:.0f} cm  (arm {arm*100:.1f} cm, props {prop_radius*100:.1f} cm radius)")
    print(f"  thrust:weight   {t2w}   -> {t2w*mass*1000:.0f} g of thrust, {(t2w-1)*9.81:.1f} m/s2 spare accel")
    print(f"  hover RPM       {hover_rpm:.0f} of {max_rpm:.0f} max ({100*hover_rpm/max_rpm:.0f}% -- headroom for attitude control)")
    print(f"  inertia izz     {izz:.2e} kg m2 ({izz/(0.5*0.027*0.0397**2):.0f}x the Crazyflie: slower to rotate)")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a quadrotor URDF for gym-pybullet-drones")
    ap.add_argument("--name", required=True, help="asset name; also the DroneModel value to register")
    ap.add_argument("--mass", type=float, required=True, help="all-up mass in kg, including payload and battery")
    ap.add_argument("--wheelbase", type=float, required=True, help="motor-to-motor diagonal in metres")
    ap.add_argument("--thrust2weight", type=float, default=2.2, help="total static thrust / weight; 2.0-2.5 is typical for a loaded carrier, 4+ for a racer")
    ap.add_argument("--max-speed-kmh", type=float, default=60.0)
    ap.add_argument("--prop-radius", type=float, default=None)
    args = ap.parse_args()
    build(args.name, args.mass, args.wheelbase, args.thrust2weight, args.max_speed_kmh, args.prop_radius)
