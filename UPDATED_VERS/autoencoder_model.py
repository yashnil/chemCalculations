#!/usr/bin/env python3
"""
Flow-map Autoencoder (FlowMapAutoencoder)
=========================================

This module implements the FlowMapAutoencoder architecture shared by the user.
It is a self-contained PyTorch module that exposes a factory helper
(`create_model`) as well as the individual building blocks (Encoder,
LatentDynamics, Decoder).

The code is adapted verbatim from the user's reference `model.py`, with only
minor doc / typing tweaks to make it easier to import inside the new
`UPDATED_VERS` toolkit.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

ACTIVATION_ALIASES = {
    "leakyrelu": "LeakyReLU",
    "leaky_relu": "LeakyReLU",
    "relu": "ReLU",
    "gelu": "GELU",
    "silu": "SiLU",
    "swish": "SiLU",
    "tanh": "Tanh",
    "elu": "ELU",
}


def get_activation(name: str) -> nn.Module:
    """Factory with in-place fast-paths where safe."""
    n = name.lower()
    cls_name = ACTIVATION_ALIASES.get(n, name)
    if cls_name == "LeakyReLU":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if cls_name == "ReLU":
        return nn.ReLU(inplace=True)
    if cls_name == "SiLU":
        return nn.SiLU(inplace=True)
    if cls_name == "ELU":
        return nn.ELU(inplace=True)
    act = getattr(nn, cls_name, None)
    if act is None or not issubclass(act, nn.Module):
        supported = sorted(set(ACTIVATION_ALIASES.keys()) | {"ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "ELU"})
        raise ValueError(f"Unknown activation '{name}'. Supported: {', '.join(supported)}")
    return act()


def _fresh_activation(template: nn.Module) -> nn.Module:
    """Clone an activation of the same kind, preferring in-place variants."""
    c = template.__class__.__name__.lower()
    if c == "leakyrelu":
        slope = getattr(template, "negative_slope", 0.01)
        return nn.LeakyReLU(negative_slope=float(slope), inplace=True)
    if c == "relu":
        return nn.ReLU(inplace=True)
    if c == "silu":
        return nn.SiLU(inplace=True)
    if c == "elu":
        return nn.ELU(inplace=True)
    return template.__class__()  # GELU/Tanh/etc.


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple multi-layer perceptron that supports broadcasting on leading dims."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: nn.Module,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(_fresh_activation(activation))
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Encoder(nn.Module):
    """Encoder: concatenates state and global context -> latent z."""

    def __init__(
        self,
        state_dim: int,
        global_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        activation: nn.Module,
        dropout_p: float = 0.0,
        vae_mode: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.global_dim = global_dim
        self.latent_dim = latent_dim
        self.vae_mode = bool(vae_mode)

        inp = state_dim + global_dim
        out = latent_dim * 2 if self.vae_mode else latent_dim

        self.network = MLP(
            input_dim=inp,
            hidden_dims=hidden_dims,
            output_dim=out,
            activation=activation,
            dropout_p=dropout_p,
        )

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.cat([y, g], dim=-1)
        out = self.network(x)

        if not self.vae_mode:
            return out, None

        mu, logvar = torch.chunk(out, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl


class LatentDynamics(nn.Module):
    """Latent dynamics: evolves z over dt with optional residual connection."""

    def __init__(
        self,
        latent_dim: int,
        global_dim: int,
        hidden_dims: Sequence[int],
        activation: nn.Module,
        dropout_p: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.global_dim = global_dim
        self.residual = bool(residual)

        inp = latent_dim + 1 + global_dim
        out = latent_dim

        self.network = MLP(
            input_dim=inp,
            hidden_dims=hidden_dims,
            output_dim=out,
            activation=activation,
            dropout_p=dropout_p,
        )

        if self.residual:
            with torch.no_grad():
                lin_out: nn.Linear = self.network.network[-1]  # type: ignore[assignment]
                nn.init.zeros_(lin_out.bias)
                lin_out.weight.mul_(0.1)

    def forward(self, z: torch.Tensor, dt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if dt.ndim == 1:
            dt = dt.unsqueeze(1)
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)
        B, K = z.shape[0], dt.shape[1]

        z_exp = z.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)
        x = torch.cat([z_exp, dt.unsqueeze(-1), g_exp], dim=-1)

        dz = self.network(x)
        return (z_exp + dz) if self.residual else dz


class Decoder(nn.Module):
    """Decoder: latent z (broadcast over K) -> reconstruction."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int],
        state_dim: int,
        activation: nn.Module,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.network = MLP(
            input_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=state_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class FlowMapAutoencoder(nn.Module):
    """Flow-map AE: (y_i, g, dt_norm) -> y_j in latent space."""

    def __init__(
        self,
        *,
        state_dim_in: int,
        state_dim_out: int,
        global_dim: int,
        latent_dim: int,
        encoder_hidden: Sequence[int],
        dynamics_hidden: Sequence[int],
        decoder_hidden: Sequence[int],
        activation_name: str = "gelu",
        dropout: float = 0.0,
        vae_mode: bool = False,
        dynamics_residual: bool = True,
        predict_delta: bool = True,
        predict_delta_log_phys: bool = False,
        target_idx: Optional[torch.Tensor] = None,
        target_log_mean: Optional[Sequence[float]] = None,
        target_log_std: Optional[Sequence[float]] = None,
        softmax_head: bool = False,
        allow_partial_simplex: bool = False,
    ):
        super().__init__()
        self.S_in = int(state_dim_in)
        self.S_out = int(state_dim_out)
        self.G = int(global_dim)
        self.Z = int(latent_dim)

        self.vae_mode = bool(vae_mode)
        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)
        self.softmax_head = bool(softmax_head)
        self.allow_partial_simplex = bool(allow_partial_simplex)

        if self.softmax_head and (self.predict_delta or self.predict_delta_log_phys):
            raise RuntimeError("softmax_head=True is incompatible with residual modes")
        if self.softmax_head and (self.S_out != self.S_in) and not self.allow_partial_simplex:
            raise RuntimeError("softmax_head=True with S_out!=S_in requires allow_partial_simplex=True")

        if target_idx is None:
            self.target_idx = None
        else:
            if not isinstance(target_idx, torch.Tensor):
                target_idx = torch.tensor(target_idx, dtype=torch.long)
            self.register_buffer("target_idx", target_idx, persistent=True)

        if self.predict_delta_log_phys or self.softmax_head:
            if target_log_mean is None or target_log_std is None:
                raise ValueError("target_log_mean/std required for log-phys or softmax head")
            log_mean = torch.as_tensor(target_log_mean, dtype=torch.float32)
            log_std = torch.clamp(torch.as_tensor(target_log_std, dtype=torch.float32), min=1e-10)
            self.register_buffer("log_mean", log_mean, persistent=True)
            self.register_buffer("log_std", log_std, persistent=True)
            self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32), persistent=True)
            self._logsoftmax = nn.LogSoftmax(dim=-1) if self.softmax_head else None
        else:
            self.log_mean = None  # type: ignore[assignment]
            self.log_std = None   # type: ignore[assignment]
            self.ln10 = None      # type: ignore[assignment]
            self._logsoftmax = None

        act = get_activation(activation_name)

        self.encoder = Encoder(
            state_dim=self.S_in,
            global_dim=self.G,
            hidden_dims=list(encoder_hidden),
            latent_dim=self.Z,
            activation=act,
            dropout_p=float(dropout),
            vae_mode=self.vae_mode,
        )
        self.dynamics = LatentDynamics(
            latent_dim=self.Z,
            global_dim=self.G,
            hidden_dims=list(dynamics_hidden),
            activation=act,
            dropout_p=float(dropout),
            residual=dynamics_residual,
        )
        self.decoder = Decoder(
            latent_dim=self.Z,
            hidden_dims=list(decoder_hidden),
            state_dim=self.S_out,
            activation=act,
            dropout_p=float(dropout),
        )

        if self.predict_delta or self.predict_delta_log_phys:
            with torch.no_grad():
                lin_out: nn.Linear = self.decoder.network.network[-1]  # type: ignore[assignment]
                nn.init.zeros_(lin_out.bias)
                lin_out.weight.mul_(0.1)

        self.kl_loss: Optional[torch.Tensor] = None

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        z_i, self.kl_loss = self.encoder(y_i, g)
        z_j = self.dynamics(z_i, dt_norm, g)
        y_pred = self.decoder(z_j)

        if self.softmax_head:
            log_p = self._logsoftmax(y_pred)  # type: ignore[operator]
            dtype = log_p.dtype
            ln10 = self.ln10.to(dtype=dtype)
            log10_p = log_p / ln10
            y_pred = (log10_p - self.log_mean.to(dtype)) / self.log_std.to(dtype)
            return y_pred

        if self.predict_delta_log_phys:
            if self.S_out != self.S_in:
                if not isinstance(self.target_idx, torch.Tensor):
                    raise RuntimeError("target_idx required when S_out != S_in")
                base_z = y_i.index_select(1, self.target_idx)
            else:
                base_z = y_i
            lm = self.log_mean.to(base_z.dtype)  # type: ignore[union-attr]
            ls = self.log_std.to(base_z.dtype)   # type: ignore[union-attr]
            base_log = base_z * ls + lm
            y_pred_log = base_log.unsqueeze(1) + y_pred
            y_pred = (y_pred_log - lm.to(y_pred.dtype)) / ls.to(y_pred.dtype)
            return y_pred

        if self.predict_delta:
            if self.S_out != self.S_in:
                if not isinstance(self.target_idx, torch.Tensor):
                    raise RuntimeError("target_idx required when S_out != S_in")
                base = y_i.index_select(1, self.target_idx)
            else:
                base = y_i
            y_pred = y_pred + base.unsqueeze(1)
            return y_pred

        return y_pred

    @torch.no_grad()
    def check_stat_consistency(self, loss_log_mean: torch.Tensor, loss_log_std: torch.Tensor) -> None:
        if not (self.softmax_head or self.predict_delta_log_phys):
            return
        if self.log_mean is None or self.log_std is None:
            raise RuntimeError("Model requires stats but buffers missing")
        m_mu = self.log_mean.detach().cpu().reshape(-1)
        m_sd = self.log_std.detach().cpu().reshape(-1)
        l_mu = loss_log_mean.detach().cpu().reshape(-1)
        l_sd = loss_log_std.detach().cpu().reshape(-1)
        rtol, atol = 1e-6, 1e-9
        if not torch.allclose(m_mu, l_mu, rtol=rtol, atol=atol):
            i = int((m_mu - l_mu).abs().argmax())
            raise ValueError(f"log_mean mismatch at {i}")
        if not torch.allclose(m_sd, l_sd, rtol=rtol, atol=atol):
            i = int((m_sd - l_sd).abs().argmax())
            raise ValueError(f"log_std mismatch at {i}")


def create_model(config: Dict[str, Any], logger: Optional["logging.Logger"] = None) -> FlowMapAutoencoder:
    """
    Convenience factory that mirrors the interface from the user's reference
    implementation.  Expects a dict with the following structure:

        {
            "data": {
                "species_variables": [...],
                "global_variables": [...],
                "target_species": [...],  # optional, defaults to species_variables
            },
            "model": {
                "latent_dim": 32,
                "encoder_hidden": [256, 128],
                ...
            },
            "paths": {
                "processed_data_dir": "...",  # required if using log-phys / simplex heads
            }
        }
    """
    import json
    import logging
    from pathlib import Path

    log = logger if logger is not None else logging.getLogger(__name__)

    data_cfg = config.get("data", {})
    species_vars = list(data_cfg.get("species_variables") or [])
    global_vars = list(data_cfg.get("global_variables", []))
    if not species_vars:
        raise KeyError("data.species_variables must be set and non-empty")

    target_vars = list(data_cfg.get("target_species") or species_vars)

    name_to_idx = {n: i for i, n in enumerate(species_vars)}
    try:
        target_idx = [name_to_idx[n] for n in target_vars]
    except KeyError as e:
        raise KeyError(f"target_species contains unknown name: {e.args[0]!r}")

    S_in = len(species_vars)
    S_out = len(target_vars)
    G = len(global_vars)

    mcfg = config.get("model", {})

    latent_dim = int(mcfg.get("latent_dim", 32))
    encoder_hidden = list(mcfg.get("encoder_hidden", [256, 128]))
    dynamics_hidden = list(mcfg.get("dynamics_hidden", [256, 256]))
    decoder_hidden = list(mcfg.get("decoder_hidden", [128, 256]))
    vae_mode = bool(mcfg.get("vae_mode", False))
    dynamics_residual = bool(mcfg.get("dynamics_residual", True))

    activation = str(mcfg.get("activation", "gelu"))
    dropout = float(mcfg.get("dropout", 0.0))

    predict_delta = bool(mcfg.get("predict_delta", True))
    predict_delta_log_phys = bool(mcfg.get("predict_delta_log_phys", False))
    softmax_head = bool(mcfg.get("softmax_head", False))
    allow_partial_simplex = bool(mcfg.get("allow_partial_simplex", False))

    if softmax_head:
        if predict_delta or predict_delta_log_phys:
            raise ValueError("softmax_head=True requires residual modes to be disabled")
        if S_out != S_in and not allow_partial_simplex:
            raise ValueError("softmax_head=True with subset species requires allow_partial_simplex=True")

    need_stats = predict_delta_log_phys or softmax_head
    target_log_mean = None
    target_log_std = None
    if need_stats:
        norm_path = Path(config["paths"]["processed_data_dir"]) / "normalization.json"
        with open(norm_path, "r") as f:
            manifest = json.load(f)
        stats = manifest["per_key_stats"]
        target_log_mean, target_log_std = [], []
        for name in target_vars:
            if name not in stats:
                raise KeyError(f"Target species '{name}' not in normalization stats")
            s = stats[name]
            target_log_mean.append(float(s.get("log_mean", 0.0)))
            target_log_std.append(float(s.get("log_std", 1.0)))
        log.info(f"Loaded normalization stats for {len(target_vars)} species")

    return FlowMapAutoencoder(
        state_dim_in=S_in,
        state_dim_out=S_out,
        global_dim=G,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        dynamics_hidden=dynamics_hidden,
        decoder_hidden=decoder_hidden,
        activation_name=activation,
        dropout=dropout,
        vae_mode=vae_mode,
        dynamics_residual=dynamics_residual,
        predict_delta=predict_delta,
        predict_delta_log_phys=predict_delta_log_phys,
        target_idx=torch.tensor(target_idx, dtype=torch.long),
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
        softmax_head=softmax_head,
        allow_partial_simplex=allow_partial_simplex,
    )

