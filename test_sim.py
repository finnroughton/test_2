import pybullet as p
import pybullet_data
import time
import math

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

planeId = p.loadURDF("plane.urdf")
p.changeDynamics(planeId, -1, lateralFriction=1.0)

startPos = [0, 0, 0.5]
boxId = p.loadURDF("cube.urdf", startPos)
p.changeDynamics(boxId, -1, lateralFriction=1.0)

mass = p.getDynamicsInfo(boxId, -1)[0]
gravity_force = mass * 9.8

P_gain = 20.0
D_gain = 10.0

radius = 2.0        # how wide the circle is
height = 2.0         # constant altitude while circling
speed = 2.0          # how fast it moves around the circle

for i in range(6000):
    t = i / 240.0  # elapsed simulation time in seconds

    # target moves in a circle over time
    target = [
        radius * math.cos(speed * t),
        radius * math.sin(speed * t),
        height
    ]

    pos, orn = p.getBasePositionAndOrientation(boxId)
    linVel, angVel = p.getBaseVelocity(boxId)

    forces = []
    for axis in range(3):
        error = target[axis] - pos[axis]
        velocity = linVel[axis]
        force = (error * P_gain) - (velocity * D_gain)
        if axis == 2:
            force += gravity_force
        forces.append(force)

    p.applyExternalForce(
        boxId,
        -1,
        forceObj=forces,
        posObj=[0, 0, 0],
        flags=p.LINK_FRAME
    )

    p.stepSimulation()

    if i % 60 == 0:
        print("pos:", pos, " target:", target, flush=True)

    time.sleep(1./240.)

p.disconnect()
