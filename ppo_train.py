import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.parallel import DistributedDataParallel as DDP

from phys import GRAVITY, LENGTH, MASS, phys_upd


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EPS = 1e-6
_REWARD_WEIGHT_CACHE = {}


@dataclass
class Config:
    total_steps: int = 100
    num_envs: int = 2048
    horizon: int = 512
    update_epochs: int = 4
    minibatch_size: int = 16384
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    max_torque: float = 50.0
    max_ep_len: int = 1024
    max_ep_len_jitter: float = 0.1
    init_angle_scale: float = 0.05
    init_vel_scale: float = 0.05
    seed: int = 1
    save_every: int = 50
    ckpt_path: str = "ppo_ckpt.pt"
    torch_compile: bool = False
    compile_mode: str = "reduce-overhead"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="PPO trainer for 4-link inverted pendulum"
    )
    parser.add_argument(
        "--total-steps", "--total_steps", type=int, default=Config.total_steps
    )
    parser.add_argument("--num-envs", type=int, default=Config.num_envs)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--update-epochs", type=int, default=Config.update_epochs)
    parser.add_argument("--minibatch-size", type=int, default=Config.minibatch_size)
    parser.add_argument("--gamma", type=float, default=Config.gamma)
    parser.add_argument("--gae-lambda", type=float, default=Config.gae_lambda)
    parser.add_argument("--clip-coef", type=float, default=Config.clip_coef)
    parser.add_argument("--ent-coef", type=float, default=Config.ent_coef)
    parser.add_argument("--vf-coef", type=float, default=Config.vf_coef)
    parser.add_argument("--max-grad-norm", type=float, default=Config.max_grad_norm)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--max-torque", type=float, default=Config.max_torque)
    parser.add_argument("--max-ep-len", type=int, default=Config.max_ep_len)
    parser.add_argument(
        "--max-ep-len-jitter",
        type=float,
        default=Config.max_ep_len_jitter,
        help="Episode length jitter ratio in [0, 1).",
    )
    parser.add_argument(
        "--init-angle-scale", type=float, default=Config.init_angle_scale
    )
    parser.add_argument("--init-vel-scale", type=float, default=Config.init_vel_scale)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--save-every", type=int, default=Config.save_every)
    parser.add_argument("--ckpt-path", type=str, default=Config.ckpt_path)
    parser.add_argument(
        "--torch-compile", action="store_true", default=Config.torch_compile
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        nargs="?",
        const=Config.compile_mode,
        default=Config.compile_mode,
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    args = parser.parse_args()
    return Config(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        horizon=args.horizon,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        max_torque=args.max_torque,
        max_ep_len=args.max_ep_len,
        max_ep_len_jitter=args.max_ep_len_jitter,
        init_angle_scale=args.init_angle_scale,
        init_vel_scale=args.init_vel_scale,
        seed=args.seed,
        save_every=args.save_every,
        ckpt_path=args.ckpt_path,
        torch_compile=args.torch_compile,
        compile_mode=args.compile_mode,
    )


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return distributed, rank, world_size, device


def cleanup_distributed(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def distributed_mean(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if not distributed:
        return value
    out = value.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= dist.get_world_size()
    return out


def reset_env(
    num_envs: int, device: torch.device, angle_scale: float, vel_scale: float
) -> torch.Tensor:
    q = angle_scale * torch.randn(num_envs, 4, device=device)
    dq = vel_scale * torch.randn(num_envs, 4, device=device)
    return torch.cat((q, dq), dim=-1)


def sample_episode_limits(
    num_envs: int,
    base_ep_len: int,
    jitter_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    if base_ep_len < 1:
        raise ValueError("max_ep_len must be >= 1")
    if not 0.0 <= jitter_ratio < 1.0:
        raise ValueError("max_ep_len_jitter must be in [0, 1)")

    jitter_steps = int(base_ep_len * jitter_ratio)
    if jitter_steps == 0:
        return torch.full((num_envs,), base_ep_len, device=device, dtype=torch.long)

    low = max(1, base_ep_len - jitter_steps)
    high = base_ep_len + jitter_steps
    return torch.randint(low, high + 1, (num_envs,), device=device, dtype=torch.long)


def _reward_weight(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device.type, device.index, dtype)
    cached = _REWARD_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    mass = torch.tensor(MASS, device=device, dtype=dtype)
    length = torch.tensor(LENGTH, device=device, dtype=dtype)
    tail_mass = torch.flip(torch.cumsum(torch.flip(mass, dims=(0,)), dim=0), dims=(0,))
    weight = GRAVITY * tail_mass * length
    _REWARD_WEIGHT_CACHE[key] = weight
    return weight


def compute_reward(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
) -> torch.Tensor:
    _ = state, action
    q = next_state[..., :4]
    weight = _reward_weight(next_state.device, next_state.dtype)
    potential = torch.sum(weight * torch.cos(q), dim=-1)
    return potential


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(8, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(8, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, obs: torch.Tensor):
        mean = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, value, log_std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean, _, log_std = self.forward(obs)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value, _ = self.forward(obs)
        return value

    def sample_action(self, obs: torch.Tensor, max_torque: float):
        distn = self.get_dist(obs)
        pre_tanh = distn.rsample()
        action = torch.tanh(pre_tanh) * max_torque
        log_prob = squash_log_prob(distn, pre_tanh, max_torque)
        value = self.value(obs)
        return action, pre_tanh, log_prob, value

    def evaluate_pre_tanh(
        self, obs: torch.Tensor, pre_tanh: torch.Tensor, max_torque: float
    ):
        distn = self.get_dist(obs)
        log_prob = squash_log_prob(distn, pre_tanh, max_torque)
        entropy = distn.entropy().sum(dim=-1)
        value = self.value(obs)
        return log_prob, entropy, value


def squash_log_prob(
    distn: Normal, pre_tanh: torch.Tensor, max_torque: float
) -> torch.Tensor:
    action_unit = torch.tanh(pre_tanh)
    logp_u = distn.log_prob(pre_tanh).sum(dim=-1)
    log_det = torch.log(max_torque * (1.0 - action_unit.pow(2)) + EPS).sum(dim=-1)
    return logp_u - log_det


def compute_gae(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    horizon = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)

    for t in reversed(range(horizon)):
        if t == horizon - 1:
            next_nonterminal = 1.0 - terminated[t]
            next_values = next_value
        else:
            next_nonterminal = 1.0 - terminated[t]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def maybe_compile(fn, enabled: bool, mode: str, rank: int, name: str):
    if not enabled or not hasattr(torch, "compile"):
        return fn
    try:
        return torch.compile(fn, mode=mode)
    except Exception as exc:
        if rank == 0:
            print(f"compile disabled for {name}: {exc}")
        return fn


def main():
    cfg = parse_args()
    distributed, rank, world_size, device = setup_distributed()

    torch.manual_seed(cfg.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed + rank)

    local_envs = cfg.num_envs // world_size
    if local_envs * world_size != cfg.num_envs:
        raise ValueError("num_envs must be divisible by world_size")

    model = ActorCritic().to(device)
    model = maybe_compile(
        model, cfg.torch_compile, cfg.compile_mode, rank, "ActorCritic"
    )
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    state = reset_env(local_envs, device, cfg.init_angle_scale, cfg.init_vel_scale)
    ep_step = torch.zeros(local_envs, device=device, dtype=torch.long)
    ep_limit = sample_episode_limits(
        local_envs,
        cfg.max_ep_len,
        cfg.max_ep_len_jitter,
        device,
    )

    steps_per_update = cfg.horizon * cfg.num_envs
    num_updates = cfg.total_steps
    if num_updates < 1:
        raise ValueError("total_steps is too small for one PPO update")

    obs_buf = torch.empty(cfg.horizon, local_envs, 8, device=device)
    pre_tanh_buf = torch.empty(cfg.horizon, local_envs, 1, device=device)
    logp_buf = torch.empty(cfg.horizon, local_envs, device=device)
    rew_buf = torch.empty(cfg.horizon, local_envs, device=device)
    terminated_buf = torch.empty(cfg.horizon, local_envs, device=device)
    val_buf = torch.empty(cfg.horizon, local_envs, device=device)

    model_rollout = model.module if isinstance(model, DDP) else model
    phys_step = maybe_compile(
        phys_upd, cfg.torch_compile, cfg.compile_mode, rank, "phys_upd"
    )

    train_start_time = time.perf_counter()

    for update in range(1, num_updates + 1):
        update_start_time = time.perf_counter()
        for t in range(cfg.horizon):
            obs_buf[t] = state
            with torch.no_grad():
                action, pre_tanh, logp, value = model_rollout.sample_action(
                    state, cfg.max_torque
                )

            next_state = phys_step(state, action.squeeze(-1))
            reward = compute_reward(state, action, next_state)

            terminated = ~torch.isfinite(next_state).all(dim=-1)
            ep_step = ep_step + 1
            truncated = ep_step >= ep_limit
            done = terminated | truncated

            pre_tanh_buf[t] = pre_tanh
            logp_buf[t] = logp
            rew_buf[t] = reward
            terminated_buf[t] = terminated.to(dtype=reward.dtype)
            val_buf[t] = value

            reset_state = reset_env(
                local_envs,
                device,
                cfg.init_angle_scale,
                cfg.init_vel_scale,
            )
            reset_ep_limit = sample_episode_limits(
                local_envs,
                cfg.max_ep_len,
                cfg.max_ep_len_jitter,
                device,
            )
            done_mask = done.unsqueeze(-1)
            next_state = torch.where(done_mask, reset_state, next_state)
            ep_step = torch.where(done, torch.zeros_like(ep_step), ep_step)
            ep_limit = torch.where(done, reset_ep_limit, ep_limit)

            state = next_state

        with torch.no_grad():
            next_value = model_rollout.value(state)

        adv_buf, ret_buf = compute_gae(
            rew_buf,
            terminated_buf,
            val_buf,
            next_value,
            cfg.gamma,
            cfg.gae_lambda,
        )

        b_obs = obs_buf.reshape(-1, 8)
        b_pre_tanh = pre_tanh_buf.reshape(-1, 1)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv_buf.reshape(-1)
        b_ret = ret_buf.reshape(-1)
        b_val = val_buf.reshape(-1)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1e-8)

        batch_size = b_obs.shape[0]
        if cfg.minibatch_size > batch_size:
            raise ValueError("minibatch_size must be <= horizon * local_envs")

        clipfracs = []
        approx_kls = []

        for _ in range(cfg.update_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, cfg.minibatch_size):
                mb_idx = indices[start : start + cfg.minibatch_size]

                mean, new_value, log_std = model(b_obs[mb_idx])
                std = torch.exp(log_std).expand_as(mean)
                distn = Normal(mean, std)
                new_logp = squash_log_prob(distn, b_pre_tanh[mb_idx], cfg.max_torque)
                entropy = distn.entropy().sum(dim=-1)
                log_ratio = new_logp - b_logp[mb_idx]
                ratio = torch.exp(log_ratio)

                mb_adv = b_adv[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (new_value - b_ret[mb_idx]).pow(2)
                v_clipped = b_val[mb_idx] + torch.clamp(
                    new_value - b_val[mb_idx], -cfg.clip_coef, cfg.clip_coef
                )
                v_loss_clipped = (v_clipped - b_ret[mb_idx]).pow(2)
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (b_logp[mb_idx] - new_logp).mean()
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
                    approx_kls.append(approx_kl)
                    clipfracs.append(clipfrac)

        with torch.no_grad():
            mean_rew = rew_buf.mean()
            mean_kl = (
                torch.stack(approx_kls).mean()
                if approx_kls
                else torch.tensor(0.0, device=device)
            )
            mean_clipfrac = (
                torch.stack(clipfracs).mean()
                if clipfracs
                else torch.tensor(0.0, device=device)
            )

            mean_rew = distributed_mean(mean_rew, distributed)
            mean_kl = distributed_mean(mean_kl, distributed)
            mean_clipfrac = distributed_mean(mean_clipfrac, distributed)

        if rank == 0:
            global_step = update * steps_per_update
            update_time_s = time.perf_counter() - update_start_time
            total_time_s = time.perf_counter() - train_start_time
            sps = steps_per_update / max(update_time_s, 1e-8)
            print(
                f"update={update}/{num_updates} global_step={global_step} "
                f"mean_rew={mean_rew.item():.4f} kl={mean_kl.item():.6f} clipfrac={mean_clipfrac.item():.4f} "
                f"update_time_s={update_time_s:.3f} total_time_s={total_time_s:.3f} sps={sps:.1f}"
            )

            if update % cfg.save_every == 0 or update == num_updates:
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save(
                    {
                        "model": model_to_save.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": cfg.__dict__,
                        "update": update,
                    },
                    cfg.ckpt_path,
                )

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
