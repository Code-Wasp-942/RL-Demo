import torch


# Physical parameters (edit here if needed)
N_LINKS = 4
MASS = (1.0, 1.0, 1.0, 1.0)
LENGTH = (1.0, 1.0, 1.0, 1.0)
GRAVITY = 9.81
DT = 0.01
DAMPING = 0


_STATIC_CACHE = {}


def _constants(device: torch.device, dtype: torch.dtype):
    key = (device.type, device.index, dtype)
    cached = _STATIC_CACHE.get(key)
    if cached is not None:
        return cached

    mass = torch.tensor(MASS, device=device, dtype=dtype)
    length = torch.tensor(LENGTH, device=device, dtype=dtype)
    tail_mass = torch.flip(torch.cumsum(torch.flip(mass, dims=(0,)), dim=0), dims=(0,))

    idx = torch.arange(N_LINKS, device=device)
    ii, jj = torch.meshgrid(idx, idx, indexing="ij")
    li_lj = length[ii] * length[jj]
    pair_mass = tail_mass[torch.maximum(ii, jj)]
    weight = li_lj * pair_mass

    coeff_stack = []
    for k in range(N_LINKS):
        coeff = (jj == k).to(dtype=dtype) - (ii == k).to(dtype=dtype)
        coeff_stack.append(coeff)
    coeff_stack = torch.stack(coeff_stack, dim=0)

    gravity_weight = tail_mass * length * GRAVITY
    pi = torch.tensor(torch.pi, device=device, dtype=dtype)
    two_pi = 2.0 * pi

    cached = (gravity_weight, weight, coeff_stack, pi, two_pi)
    _STATIC_CACHE[key] = cached
    return cached


def _lagrangian_accel(q: torch.Tensor, dq: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
    """Compute qdd for a 4-link serial pendulum using Lagrangian dynamics."""
    device = q.device
    dtype = q.dtype

    gravity_weight, weight, coeff_stack, _, _ = _constants(device, dtype)

    q_diff = q[..., :, None] - q[..., None, :]
    cos_diff = torch.cos(q_diff)
    sin_diff = torch.sin(q_diff)

    mass_matrix = weight * cos_diff

    # dM/dq_k tensor: (..., i, j, k)
    d_mass = torch.einsum("...ij,kij->...ijk", weight * sin_diff, coeff_stack)

    # Christoffel symbols Γ_ijk = 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
    d_ikj = d_mass.transpose(-2, -1)
    d_jki = d_mass.permute(*range(d_mass.ndim - 3), d_mass.ndim - 2, d_mass.ndim - 1, d_mass.ndim - 3)
    gamma = 0.5 * (d_mass + d_ikj - d_jki)

    coriolis = torch.einsum("...ijk,...j,...k->...i", gamma, dq, dq)

    gravity = -gravity_weight * torch.sin(q)
    damping = DAMPING * dq

    rhs = -coriolis - gravity - damping
    rhs[..., 0] = rhs[..., 0] + op
    qdd = torch.linalg.solve(mass_matrix, rhs.unsqueeze(-1)).squeeze(-1)
    return qdd


def phys_upd(state: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
    """
    One-step dynamics update for 4-link inverted pendulum.

    Args:
        state: tensor with shape (..., 8), [q1..q4, dq1..dq4]. q=0 means upward.
        op: torque applied at joint 1 (bottom rod), shape broadcastable to (...,) or (...,1).

    Returns:
        next_state: tensor with shape (..., 8)
    """
    if state.shape[-1] != 2 * N_LINKS:
        raise ValueError(f"Expected state[..., {2 * N_LINKS}], got {tuple(state.shape)}")

    q = state[..., :N_LINKS]
    dq = state[..., N_LINKS:]

    op = torch.as_tensor(op, device=state.device, dtype=state.dtype)
    if op.ndim == q.ndim and op.shape[-1] == 1:
        op = op.squeeze(-1)

    qdd = _lagrangian_accel(q, dq, op)

    # Semi-implicit Euler
    dq_next = dq + DT * qdd
    q_next = q + DT * dq_next

    # Angle normalization to [-pi, pi) using mod
    _, _, _, pi, two_pi = _constants(state.device, state.dtype)
    q_next = torch.remainder(q_next + pi, two_pi) - pi

    return torch.cat((q_next, dq_next), dim=-1)
