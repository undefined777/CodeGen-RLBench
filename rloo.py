# RLOO (REINFORCE Leave-One-Out) Implementation
# Simplified version: only supports chunked forward processing

__all__ = ["RLOOConfig", "RLOOTrainer", "create_rloo_trainer"]

import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
from torch.optim import AdamW

from utils import logprobs_from_logits


@dataclass
class RLOOConfig:
    """RLOO configuration parameters"""
    # Learning parameters
    lr: float = 1e-6
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    
    # RLOO specific parameters
    group_size: int = 4              # Number of candidates per question G
    cliprange: float = 0.2           # PPO clip (optional)
    
    # KL divergence control
    kl_coef: float = 0.04
    kl_mode: str = "forward"        # "forward" or "unbiased"
    
    # Chunked processing
    train_micro_batch_size: int = 2  # Chunk along sample dimension (N=B*G)
    
    # Device configuration
    amp_dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    
    # RLOO specific configuration
    use_clipping: bool = False       # RLOO typically doesn't use PPO clipping
    baseline_mode: str = "leave_one_out"  # "leave_one_out" or "mean"
    
    # Others
    tokenizer = None


class RLOOTrainer:
    """RLOO Trainer - Based on REINFORCE Leave-One-Out"""

    def __init__(self, model, ref_model, config: Optional[RLOOConfig] = None, **kwargs):
        self.config = config or RLOOConfig()
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        
        self.model = model
        self.ref_model = ref_model
        # Freeze reference model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.config.lr, 
            eps=self.config.adam_eps, 
            weight_decay=self.config.weight_decay
        )

        self.logger = logging.getLogger(__name__)
        self.step_count = 0

    # ---------------------------- RLOO Core Algorithm ----------------------------
    def compute_leave_one_out_advantages(
        self,
        rewards: torch.Tensor,           # [N, T] or [N], N=B*G
        group_size: int,
        mask: Optional[torch.Tensor] = None  # [N, T] optional
    ) -> torch.Tensor:
        """
        RLOO core: Leave-One-Out baseline calculation
        For each sample, use the average reward of other samples in the same group as baseline
        """
        with torch.no_grad():
            # Convert to scalar rewards
            if rewards.dim() == 2:
                if mask is not None:
                    # Use mask weighted sum
                    flat = (rewards.float() * mask.float()).sum(dim=1)
                else:
                    flat = rewards.float().sum(dim=1)
            else:
                flat = rewards.float()

            N = flat.numel()
            assert N % group_size == 0, "N must be divisible by group_size"
            B = N // group_size
            flat = flat.view(B, group_size)  # [B, G]

            # Leave-One-Out baseline: for each sample, baseline is the average of other samples in the same group
            advantages = torch.zeros_like(flat)
            
            for i in range(group_size):
                # For the i-th sample, calculate the average of other G-1 samples as baseline
                mask_others = torch.ones(group_size, dtype=torch.bool, device=flat.device)
                mask_others[i] = False
                
                if group_size > 1:
                    baseline = flat[:, mask_others].mean(dim=1)  # [B]
                    advantages[:, i] = flat[:, i] - baseline
                else:
                    # If only one sample, advantage is 0
                    advantages[:, i] = 0.0
            
            return advantages.view(-1).detach()  # [N]

    def compute_mean_baseline_advantages(
        self,
        rewards: torch.Tensor,           # [N, T] or [N], N=B*G
        group_size: int,
        mask: Optional[torch.Tensor] = None  # [N, T] optional
    ) -> torch.Tensor:
        """
        Alternative: use group average as baseline (similar to REINFORCE with baseline)
        """
        with torch.no_grad():
            # Convert to scalar rewards
            if rewards.dim() == 2:
                if mask is not None:
                    flat = (rewards.float() * mask.float()).sum(dim=1)
                else:
                    flat = rewards.float().sum(dim=1)
            else:
                flat = rewards.float()

            N = flat.numel()
            assert N % group_size == 0, "N must be divisible by group_size"
            B = N // group_size
            flat = flat.view(B, group_size)  # [B, G]

            # Use group average as baseline
            baseline = flat.mean(dim=1, keepdim=True)  # [B, 1]
            advantages = flat - baseline  # [B, G]
            
            return advantages.view(-1).detach()  # [N]

    @staticmethod
    def expand_advantages_to_tokens(adv_scalar: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """[N] → [N, T] (broadcast to tokens and multiply by mask)"""
        return adv_scalar.detach().unsqueeze(1) * response_mask.float()

    # ---------------------------- Inherit other methods from GRPO ----------------------------
    def forward_pass(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forcing → return log_probs (aligned with input_ids[:, 1:]), shape [B, L-1]
        Compatible with HF return_dict=True/False return formats
        """
        # Training phase doesn't store KV cache (save memory)
        if hasattr(model, "config"):
            try:
                model.config.use_cache = False
            except Exception:
                pass

        dtype = getattr(self.config, "amp_dtype", None)
        if dtype is not None:
            ctx = torch.amp.autocast(self.config.device, dtype=dtype)
        else:
            class _Dummy:
                def __enter__(self): return None
                def __exit__(self, *a): return False
            ctx = _Dummy()

        with ctx:
            # Try to request dict return; if outer wrapper ignores it, we'll handle compatibility below
            try:
                out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            except TypeError:
                out = model(input_ids=input_ids, attention_mask=attention_mask)

            # Compatible with tuple / ModelOutput
            logits = out.logits if hasattr(out, "logits") else out[0]

        # Only keep needed log_probs, immediately discard huge logits
        log_probs = logprobs_from_logits(logits[:, :-1], input_ids[:, 1:])  # [B, L-1]
        return log_probs

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,           # [N, T]
        old_log_probs: torch.Tensor,       # [N, T]
        adv_tokens: torch.Tensor,          # [N, T]
        response_masks: torch.Tensor       # [N, T]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        RLOO policy loss: optionally use or not use PPO clipping
        RLOO typically uses standard REINFORCE without clipping
        """
        mask = response_masks.float()
        
        if self.config.use_clipping:
            # Use PPO-style clipping
            ratio = torch.exp(log_probs - old_log_probs)  # [N, T]
            ratio_clip = torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)
            tok_obj = torch.minimum(ratio * adv_tokens, ratio_clip * adv_tokens) * mask
        else:
            # Standard REINFORCE: directly use log probabilities and advantages
            tok_obj = log_probs * adv_tokens * mask

        # Per-sequence average
        per_seq = tok_obj.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        policy_loss = -per_seq.mean()

        # Statistics
        with torch.no_grad():
            tok_count = mask.sum().clamp_min(1.0)
            
            if self.config.use_clipping:
                ratio = torch.exp(log_probs - old_log_probs)
                clipped = ((ratio > 1.0 + self.config.cliprange) | (ratio < 1.0 - self.config.cliprange)).float() * mask
                clipfrac = (clipped.sum() / tok_count).item()
                ratio_mean_tensor = (ratio * mask).sum() / tok_count
                ratio_mean = float(ratio_mean_tensor.item())
                ratio_variance = ((ratio - ratio_mean_tensor) ** 2 * mask).sum() / tok_count
                ratio_std = float(ratio_variance.sqrt().item()) if tok_count > 1 else 0.0
            else:
                clipfrac = 0.0
                ratio_mean = 1.0
                ratio_std = 0.0

        stats = {
            "policy/loss": float(policy_loss),
            "policy/clipfrac": clipfrac,
            "ratio/mean": ratio_mean,
            "ratio/std": ratio_std,
            "policy/clipping_enabled": float(self.config.use_clipping),
        }
        return policy_loss, stats

    def compute_kl_penalty(
        self,
        ref_logprobs: torch.Tensor,     # [N, T]  no_grad
        log_probs: torch.Tensor,        # [N, T]
        response_masks: torch.Tensor,   # [N, T]
        *,
        mode: str = "forward",         # "forward" or "unbiased"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return (β*mean_kl, stats). Same per-sequence average as policy."""
        mask = response_masks.float()
        if mode == "unbiased":
            diff = (ref_logprobs - log_probs)
            tok_kl = (torch.exp(diff) - diff - 1.0) * mask
            mode_flag = 1.0
        else:
            tok_kl = (log_probs - ref_logprobs) * mask
            mode_flag = 0.0
        per_seq = tok_kl.sum(1) / mask.sum(1).clamp_min(1.0)
        mean_kl = per_seq.mean()
        kl_penalty = self.config.kl_coef * mean_kl
        stats = {
            "kl/mean": float(mean_kl),
            "kl/penalty": float(kl_penalty),
            "kl/mode": mode_flag,
        }
        return kl_penalty, stats

    # ---------------------------- RLOO Training Step ----------------------------
    def step(
        self,
        source_ids: torch.Tensor,        # [B, S]
        source_mask: torch.Tensor,       # [B, S]
        response_ids: torch.Tensor,      # [N, T]  N=B*G
        response_ids_ref: torch.Tensor,  # unused
        rewards: torch.Tensor,           # [N, T] or [N]
        response_mask: torch.Tensor      # [N, T]
    ) -> Dict[str, float]:
        """RLOO training step - chunked forward processing"""
        self.model.train()
        t0 = time.time()
        B = source_ids.size(0)
        N, T = response_ids.shape
        G = N // B
        assert G == self.config.group_size, "Group size mismatch"

        # Concatenate prompt to each candidate
        expanded_source_ids  = source_ids.repeat_interleave(G, dim=0)
        expanded_source_mask = source_mask.repeat_interleave(G, dim=0)
        full_input_ids  = torch.cat([expanded_source_ids,  response_ids],  dim=1)
        full_attn_mask  = torch.cat([expanded_source_mask, response_mask], dim=1)

        # Compute RLOO advantages
        if self.config.baseline_mode == "leave_one_out":
            adv_scalar = self.compute_leave_one_out_advantages(rewards, G, response_mask)
        else:
            adv_scalar = self.compute_mean_baseline_advantages(rewards, G, response_mask)

        micro_N = self.config.train_micro_batch_size
        stats: Dict[str, float] = {}

        # RLOO single update
        self.optimizer.zero_grad(set_to_none=True)
        total_policy, total_kl, total_loss = 0.0, 0.0, 0.0

        # Chunked forward processing
        for start in range(0, N, micro_N):
            sl = slice(start, min(start + micro_N, N))
            n_chunk = sl.stop - sl.start
            weight = n_chunk / float(N)

            # Forward pass
            new_logp_full = self.forward_pass(self.model, full_input_ids[sl], full_attn_mask[sl])
            new_logp = new_logp_full[:, -T:]
            old_logp = new_logp.detach()  # RLOO: on-policy

            with torch.no_grad():
                ref_logp_full = self.forward_pass(self.ref_model, full_input_ids[sl], full_attn_mask[sl])
                ref_logp = ref_logp_full[:, -T:]

            # Compute advantages
            adv_tok = self.expand_advantages_to_tokens(adv_scalar[sl], response_mask[sl])

            # Compute losses
            pol_loss, pol_stats = self.compute_policy_loss(new_logp, old_logp, adv_tok, response_mask[sl])
            kl_penalty, kl_stats = self.compute_kl_penalty(ref_logp, new_logp, response_mask[sl], mode=self.config.kl_mode)

            loss = (pol_loss + kl_penalty) * weight
            loss.backward()

            total_policy += float(pol_loss.detach()) * weight
            total_kl     += float(kl_stats["kl/mean"]) * weight
            total_loss   += float((pol_loss + kl_penalty).detach()) * weight

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Compute statistics
        with torch.no_grad():
            # Basic advantage statistics
            adv_abs_mean = float(adv_scalar.abs().mean().item())
            adv_std = float(adv_scalar.std(unbiased=False).item())
            
            # Within-group differences
            B = N // G
            adv_reshaped = adv_scalar.view(B, G)
            group_ranges = adv_reshaped.max(dim=1)[0] - adv_reshaped.min(dim=1)[0]
            mean_group_range = float(group_ranges.mean().item())
            
            # Gradient norm
            total_grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

        stats.update({
            "rloo/policy_loss": total_policy,
            "rloo/mean_kl": total_kl,
            "rloo/total_loss": total_loss,
            "rloo/adv_mean": float(adv_scalar.mean().item()),
            "rloo/adv_std": adv_std,
            "rloo/adv_abs_mean": adv_abs_mean,
            "rloo/group_diversity": mean_group_range,
            "rloo/grad_norm": total_grad_norm,
            "rloo/group_size": G,
            "rloo/baseline_mode": 1.0 if self.config.baseline_mode == "leave_one_out" else 0.0,
            "rloo/use_clipping": float(self.config.use_clipping),
            "rloo/original_batch_size": B,
            "rloo/expanded_batch_size": N,
            "time/rloo/total": time.time() - t0,
        })

        # Memory cleanup
        if self.step_count % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.step_count += 1
        return stats




def create_rloo_trainer(model, ref_model, learning_rate: float = 1e-6,
                        kl_coef: float = 0.04, group_size: int = 4, 
                        use_clipping: bool = False, **kwargs) -> RLOOTrainer:
    """Convenience function to create RLOO trainer"""
    config = RLOOConfig(
        lr=learning_rate, 
        kl_coef=kl_coef, 
        group_size=group_size,
        use_clipping=use_clipping,
        **kwargs
    )
    return RLOOTrainer(model, ref_model, config)
