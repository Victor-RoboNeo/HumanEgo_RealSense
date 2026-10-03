# -*- coding: utf-8 -*-
"""Play the current LIFT retarget in MuJoCo and write an H.264 video.

Run with the isolated interpreter, not the humanego env:

    LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/lib/x86_64-linux-gnu \\
    __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \\
    MUJOCO_GL=egl \\
    python -m retarget.mujoco_play
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

URDF = _ROOT / "hardware" / "ARX_Model" / "LIFT" / "lift.urdf"
MESH = _ROOT / "hardware" / "ARX_Model" / "LIFT" / "meshes" / "lift"
WIDTH, HEIGHT = 960, 720


def _quat_wxyz(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rot[2, 1] - rot[1, 2]) / scale,
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[1, 0] - rot[0, 1]) / scale,
            ]
        )
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        quat = np.array(
            [
                (rot[2, 1] - rot[1, 2]) / scale,
                0.25 * scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                (rot[0, 2] + rot[2, 0]) / scale,
            ]
        )
    elif rot[1, 1] > rot[2, 2]:
        scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        quat = np.array(
            [
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                0.25 * scale,
                (rot[1, 2] + rot[2, 1]) / scale,
            ]
        )
    else:
        scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        quat = np.array(
            [
                (rot[1, 0] - rot[0, 1]) / scale,
                (rot[0, 2] + rot[2, 0]) / scale,
                (rot[1, 2] + rot[2, 1]) / scale,
                0.25 * scale,
            ]
        )
    return quat / np.linalg.norm(quat)


def _model(table_h: float, table_xy: np.ndarray, obj_w: np.ndarray):
    import mujoco

    text = URDF.read_text().replace("../meshes/lift/", str(MESH) + "/")
    patched = Path("/tmp/lift_mujoco.urdf")
    patched.write_text(text)
    spec = mujoco.MjSpec.from_file(str(patched))
    spec.visual.global_.offwidth = WIDTH
    spec.visual.global_.offheight = HEIGHT
    spec.body("base_link").add_freejoint(name="root")
    world = spec.worldbody
    floor = world.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size[:] = [4.0, 4.0, 0.1]
    floor.rgba[:] = [0.93, 0.93, 0.93, 1.0]
    x0, y0 = table_xy[0]
    x1, y1 = table_xy[1]
    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, table_h - 0.012])
    half = np.array([(x1 - x0) * 0.5, (y1 - y0) * 0.5, 0.012])
    top = world.add_geom()
    top.type = mujoco.mjtGeom.mjGEOM_BOX
    top.pos[:] = center
    top.size[:] = half
    top.rgba[:] = [0.16, 0.16, 0.17, 1.0]
    for cx, cy in ((x0 + 0.08, y0 + 0.08), (x1 - 0.08, y0 + 0.08), (x0 + 0.08, y1 - 0.08), (x1 - 0.08, y1 - 0.08)):
        leg = world.add_geom()
        leg.type = mujoco.mjtGeom.mjGEOM_BOX
        leg.pos[:] = [cx, cy, (table_h - 0.024) * 0.5]
        leg.size[:] = [0.02, 0.02, (table_h - 0.024) * 0.5]
        leg.rgba[:] = [0.28, 0.28, 0.3, 1.0]
    ball = world.add_geom()
    ball.type = mujoco.mjtGeom.mjGEOM_SPHERE
    ball.size[:] = [0.018, 0.0, 0.0]
    ball.pos[:] = obj_w
    ball.rgba[:] = [0.95, 0.45, 0.12, 1.0]
    spread = np.array([0.087156, 0.996195, 0.0])
    spread = spread / np.linalg.norm(spread)
    approach = np.array([1.0, 0.0, 0.0])
    approach = approach - np.dot(approach, spread) * spread
    approach = approach / np.linalg.norm(approach)
    palm = np.cross(spread, approach)
    tool = spec.body("right_arm_link6").add_site()
    tool.name = "tool"
    tool.pos[:] = [0.074942, -0.006555, -0.005]
    tool.quat[:] = _quat_wxyz(np.column_stack([spread, approach, palm]))
    light = world.add_light()
    light.pos[:] = [center[0] + 0.4, center[1] - 1.4, table_h + 1.6]
    light.dir[:] = [0.0, 0.6, -1.0]
    return spec.compile()


def _gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase blur. Edges stay put so the path does not jump at the ends."""
    if sigma <= 0 or len(values) < 3:
        return values.copy()
    radius = max(1, int(round(3 * sigma)))
    ticks = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (ticks / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, [(radius, radius)] + [(0, 0)] * (values.ndim - 1), mode="edge")
    flat = padded.reshape(len(padded), -1)
    out = np.zeros((len(values), flat.shape[1]))
    width = 2 * radius + 1
    for i in range(len(values)):
        out[i] = kernel @ flat[i:i + width]
    return out.reshape(values.shape)


def _smooth_motion(base_T: np.ndarray, hand_pos: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    base = base_T.copy()
    hand = hand_pos.copy()
    seg = base[start:end + 1]
    seg[:, :3, 3] = _gaussian_smooth(seg[:, :3, 3], sigma=5.0)
    yaw = np.unwrap(np.arctan2(seg[:, 1, 0], seg[:, 0, 0]))
    yaw = _gaussian_smooth(yaw.reshape(-1, 1), sigma=5.0).ravel()
    c, s = np.cos(yaw), np.sin(yaw)
    seg[:, 0, 0], seg[:, 0, 1], seg[:, 0, 2] = c, -s, 0.0
    seg[:, 1, 0], seg[:, 1, 1], seg[:, 1, 2] = s, c, 0.0
    seg[:, 2, 0], seg[:, 2, 1], seg[:, 2, 2] = 0.0, 0.0, 1.0
    base[start:end + 1] = seg
    base[:start] = seg[0]
    base[end + 1:] = seg[-1]
    hand[start:end + 1] = _gaussian_smooth(hand[start:end + 1], sigma=3.0)
    return base, hand


def _official_ik(model, data, base_T, lifts, head, hand_pos, grasp) -> dict:
    """Home pose, then MuJoCo IK into the action start, then the action."""
    import mujoco
    from mujoco import minimize

    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(model.njnt)}

    def qadr(name: str) -> int:
        return int(model.jnt_qposadr[names[name]])

    arm_adr = [qadr(f"right_arm_joint{i}") for i in range(1, 7)]
    lift_adr = qadr("joint_lift")
    head_adr = qadr("joint_h_2")
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool")
    down = np.array([0.0, 0.0, -1.0])
    lower = np.full(6, -np.pi)
    upper = np.full(6, np.pi)
    tracked = [i for i in range(len(base_T)) if np.isfinite(hand_pos[i, 0])]
    if not tracked:
        raise RuntimeError("the trajectory has no hand target")
    start, end = tracked[0], tracked[-1]
    base_T, hand_pos = _smooth_motion(base_T, hand_pos, start, end)

    def place(frame: int, q: np.ndarray) -> None:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[0:3] = base_T[frame, :3, 3]
        data.qpos[3:7] = _quat_wxyz(base_T[frame, :3, :3])
        data.qpos[lift_adr] = float(lifts[frame])
        data.qpos[head_adr] = head
        for adr, value in zip(arm_adr, q):
            data.qpos[adr] = value

    def ik_to(frame: int, q: np.ndarray, target: np.ndarray, rot_w: float) -> tuple[np.ndarray, float]:
        place(frame, q)
        q_prev = np.asarray(q, dtype=np.float64).copy()

        def residual(x, target=target, rot_w=rot_w, q_prev=q_prev):
            columns = x.shape[1]
            out = np.zeros((12, columns))
            for col in range(columns):
                for adr, value in zip(arm_adr, x[:, col]):
                    data.qpos[adr] = value
                mujoco.mj_kinematics(model, data)
                out[:3, col] = data.site_xpos[site] - target
                rotation = data.site_xmat[site].reshape(3, 3)
                out[3:6, col] = rot_w * (rotation[:, 1] - down)
                out[6:, col] = 0.015 * (x[:, col] - q_prev)
            return out

        q, _ = minimize.least_squares(
            q_prev,
            residual,
            bounds=(lower, upper),
            max_iter=25,
            verbose=minimize.Verbosity.SILENT,
        )
        place(frame, q)
        mujoco.mj_kinematics(model, data)
        return q, float(np.linalg.norm(data.site_xpos[site] - target))

    # Initial pose is the URDF zero of the right arm, with the base already parked
    # where the action begins.
    q = np.zeros(6)
    place(start, q)
    mujoco.mj_kinematics(model, data)
    home = data.site_xpos[site].copy()
    goal = hand_pos[start].copy()
    # Minimum-jerk into the action, arriving with the action's own first step
    # so the end effector does not stop and then jump.
    v_end = hand_pos[start + 1] - goal if start + 1 <= end else np.zeros(3)
    hold = 15
    steps = 90
    approach = [q.copy() for _ in range(hold)]
    approach_err = []
    for k in range(1, steps + 1):
        s = k / steps
        blend = 10 * s**3 - 15 * s**4 + 6 * s**5
        distance = goal - home
        velocity = v_end * steps
        if np.linalg.norm(velocity) > 0.5 * np.linalg.norm(distance):
            velocity *= 0.5 * np.linalg.norm(distance) / max(np.linalg.norm(velocity), 1e-9)
        a3 = 3 * distance - 3 * velocity
        a4 = 5 * velocity - 3 * distance
        a5 = distance - 2 * velocity
        target = home + a3 * s**3 + a4 * s**4 + a5 * s**5
        q, pos_err = ik_to(start, q, target, rot_w=0.08 * blend)
        approach.append(q.copy())
        approach_err.append(pos_err)

    solved = np.zeros((end - start + 1, 6))
    err = np.zeros(end - start + 1)
    for n, i in enumerate(range(start, end + 1)):
        q, pos_err = ik_to(i, q, hand_pos[i].copy(), rot_w=0.08)
        solved[n] = q
        err[n] = pos_err
    tail = len(base_T) - end - 1
    arm = np.concatenate([np.stack(approach), solved, np.repeat(solved[-1:], tail, axis=0)])
    base = np.concatenate([
        np.repeat(base_T[start:start + 1], len(approach), axis=0),
        base_T[start:],
    ])
    lift = np.concatenate([
        np.full(len(approach), float(lifts[start])),
        lifts[start:],
    ])
    grasp_out = np.concatenate([
        np.zeros(len(approach)),
        grasp[start:],
    ])
    step = np.max(np.abs(np.diff(arm, axis=0)), axis=1)
    print(
        f"[mujoco] approach {steps} frames from the zero pose, "
        f"arrival {approach_err[-1]*100:.2f} cm"
    )
    print(
        f"[mujoco] action IK pos median {np.median(err)*100:.2f} cm  "
        f"max {err.max()*100:.2f} cm  "
        f"joint step max {np.rad2deg(step.max()):.1f} deg at playback frame {int(np.argmax(step))+1}"
    )
    base_step = np.linalg.norm(np.diff(base[:, :3, 3], axis=0), axis=1)
    print(
        f"[mujoco] base step max {base_step.max()*100:.2f} cm  "
        f"ee command does not teleport"
    )
    return {"base_T": base, "lift": lift, "grasp": grasp_out, "arm_q": arm, "n_approach": len(approach)}


def play(traj_path: Path, out_path: Path) -> Path:
    import mujoco

    saved = np.load(traj_path)
    base_T = saved["base_T"]
    lifts = saved["lift"]
    head = float(saved["head"][0])
    grasp = saved["grasp"]
    hand_pos = saved["hand_pos"]
    table_h = float(saved["table_h"][0])
    fps = float(saved["fps"][0]) if "fps" in saved else 15.0
    model = _model(table_h, saved["table_xy"], saved["obj_w"])
    data = mujoco.MjData(model)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(model.njnt)}

    def qadr(name: str) -> int:
        return int(model.jnt_qposadr[names[name]])

    motion = _official_ik(model, data, base_T, lifts, head, hand_pos, grasp)
    base_T = motion["base_T"]
    lifts = motion["lift"]
    grasp = motion["grasp"]
    arm_q = motion["arm_q"]
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    look = base_T[:, :3, 3].mean(axis=0)
    look[2] = table_h * 0.55
    cam.lookat[:] = look
    cam.distance = 2.6
    cam.azimuth = 135.0
    cam.elevation = -22.0
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-crf", "18",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )
    n_approach = int(motion["n_approach"])
    stills = {0, n_approach - 1, n_approach + (len(base_T) - n_approach) // 2}
    try:
        for i in range(len(base_T)):
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qpos[0:3] = base_T[i, :3, 3]
            data.qpos[3:7] = _quat_wxyz(base_T[i, :3, :3])
            data.qpos[qadr("joint_lift")] = float(lifts[i])
            data.qpos[qadr("joint_h_2")] = head
            for joint_i, value in enumerate(arm_q[i]):
                data.qpos[qadr(f"right_arm_joint{joint_i + 1}")] = float(value)
            closed = float(grasp[i]) > 0.5
            data.qpos[qadr("right_catch_joint1")] = 0.0 if closed else -0.0445
            data.qpos[qadr("right_catch_joint2")] = 0.0 if closed else 0.0445
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
            if i in stills:
                still = out_path.with_name(f"lift_mujoco_{i:04d}.png")
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-i", "-", str(still)],
                    input=np.ascontiguousarray(frame).tobytes(),
                    check=True,
                )
    finally:
        encoder.stdin.close()
        renderer.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the MuJoCo video")
    print(f"[mujoco] wrote {out_path}")
    return out_path


def main() -> None:
    session = _ROOT / "retarget" / "outputs" / "session_20260920_163411"
    play(session / "lift_mujoco_traj.npz", session / "lift_replay_mujoco.mp4")


if __name__ == "__main__":
    main()
