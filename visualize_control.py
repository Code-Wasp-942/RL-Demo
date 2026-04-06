import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch

from phys import DT, LENGTH, phys_upd
from ppo_train import ActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize 4-link pendulum control")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--max-torque", type=float, default=10.0, help="Action clip bound"
    )
    parser.add_argument(
        "--torque-step", type=float, default=1.0, help="Keyboard torque step"
    )
    parser.add_argument("--fps", type=float, default=60.0, help="Rendering fps")
    parser.add_argument(
        "--sim-steps-per-frame", type=int, default=1, help="Simulation steps per frame"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reset")
    parser.add_argument(
        "--init-angle-scale", type=float, default=0.05, help="Reset angle noise scale"
    )
    parser.add_argument(
        "--init-vel-scale", type=float, default=0.05, help="Reset velocity noise scale"
    )
    parser.add_argument(
        "--ckpt", type=str, default="", help="Optional PPO checkpoint path"
    )
    parser.add_argument(
        "--autoplay", action="store_true", help="Start with PPO policy control"
    )
    return parser.parse_args()


def reset_state(
    device: torch.device,
    angle_scale: float,
    vel_scale: float,
) -> torch.Tensor:
    q = angle_scale * torch.randn(1, 4, device=device)
    dq = vel_scale * torch.randn(1, 4, device=device)
    return torch.cat((q, dq), dim=-1)


def load_policy(ckpt_path: str, device: torch.device) -> ActorCritic:
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model = ActorCritic().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def joint_positions(state: torch.Tensor):
    q = state[0, :4].detach().cpu()
    x = [0.0]
    y = [0.0]
    px = 0.0
    py = 0.0
    for length, angle in zip(LENGTH, q):
        px += float(length) * torch.sin(angle).item()
        py += float(length) * torch.cos(angle).item()
        x.append(px)
        y.append(py)
    return x, y


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    device = torch.device(args.device)

    policy = None
    if args.ckpt:
        ckpt_file = Path(args.ckpt)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")
        policy = load_policy(str(ckpt_file), device)

    state = reset_state(device, args.init_angle_scale, args.init_vel_scale)
    manual_torque = 0.0
    use_policy = bool(args.autoplay and policy is not None)
    paused = False
    torque_step = args.torque_step
    max_torque = args.max_torque

    fig, ax = plt.subplots(figsize=(7, 7))
    (arm_line,) = ax.plot([], [], "o-", lw=3, markersize=7)
    (trace_line,) = ax.plot([], [], lw=1.5, alpha=0.7)
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    total_length = float(sum(LENGTH))
    margin = 0.6
    ax.set_xlim(-(total_length + margin), total_length + margin)
    ax.set_ylim(-(total_length + margin), total_length + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("4-Link Pendulum Control")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    tip_x = []
    tip_y = []

    def toggle_policy() -> None:
        nonlocal use_policy
        if policy is None:
            print("No policy loaded. Pass --ckpt <path> first.")
            return
        use_policy = not use_policy

    def on_key(event):
        nonlocal manual_torque, state, paused, torque_step
        key = event.key
        if key == "left":
            use_manual = not use_policy
            if use_manual:
                manual_torque = max(-max_torque, manual_torque - torque_step)
        elif key == "right":
            use_manual = not use_policy
            if use_manual:
                manual_torque = min(max_torque, manual_torque + torque_step)
        elif key == "down":
            manual_torque = max(-max_torque, manual_torque - torque_step)
        elif key == "up":
            manual_torque = min(max_torque, manual_torque + torque_step)
        elif key == "0":
            manual_torque = 0.0
        elif key == "p":
            toggle_policy()
        elif key == "r":
            state = reset_state(device, args.init_angle_scale, args.init_vel_scale)
            tip_x.clear()
            tip_y.clear()
        elif key == " ":
            paused = not paused
        elif key == "[":
            torque_step = max(0.1, torque_step - 0.1)
        elif key == "]":
            torque_step = min(max_torque, torque_step + 0.1)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def frame_update(_):
        nonlocal state
        if not paused:
            for _ in range(args.sim_steps_per_frame):
                if use_policy and policy is not None:
                    with torch.no_grad():
                        action, _, _, _ = policy.sample_action(state, max_torque)
                    torque = float(action[0, 0].item())
                else:
                    torque = manual_torque
                action_tensor = torch.tensor([torque], device=device, dtype=state.dtype)
                state = phys_upd(state, action_tensor)

        xs, ys = joint_positions(state)
        arm_line.set_data(xs, ys)

        tip_x.append(xs[-1])
        tip_y.append(ys[-1])
        max_trace = int(args.fps * 10)
        if len(tip_x) > max_trace:
            del tip_x[:-max_trace]
            del tip_y[:-max_trace]
        trace_line.set_data(tip_x, tip_y)

        mode = "policy" if use_policy else "manual"
        dq = state[0, 4:].detach().cpu().tolist()
        info_text.set_text(
            f"mode: {mode}   paused: {paused}\n"
            f"torque: {manual_torque:+.2f}   step: {torque_step:.2f}   dt: {DT:.3f}\n"
            f"q: [{state[0, 0].item():+.2f}, {state[0, 1].item():+.2f}, {state[0, 2].item():+.2f}, {state[0, 3].item():+.2f}]\n"
            f"dq: [{dq[0]:+.2f}, {dq[1]:+.2f}, {dq[2]:+.2f}, {dq[3]:+.2f}]\n"
            "keys: left/right(up/down) torque, 0 reset torque, p toggle policy, r reset state, space pause"
        )
        return arm_line, trace_line, info_text

    interval_ms = max(1, int(1000.0 / args.fps))
    anim = animation.FuncAnimation(
        fig,
        frame_update,
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )
    _ = anim
    plt.show()


if __name__ == "__main__":
    main()
