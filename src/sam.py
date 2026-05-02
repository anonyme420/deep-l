"""
Sharpness-Aware Minimisation (SAM) optimiser.

SAM seeks parameters that lie in flat loss landscapes rather than sharp minima,
which improves generalisation — especially important for small medical datasets.

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
Improving Generalization", ICLR 2021. https://arxiv.org/abs/2010.01412
"""

import torch


class SAM(torch.optim.Optimizer):
    """
    SAM wrapper around any base optimiser (e.g. AdamW, SGD).

    Usage:
        base_opt = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_opt, lr=1e-4, weight_decay=1e-4)

        # Training step:
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        criterion(model(x), y).backward()
        optimizer.second_step(zero_grad=True)

    Args:
        params       : model parameters
        base_optimizer: class (not instance) of the base optimiser
        rho          : neighbourhood size (perturbation radius) — default 0.05
        **kwargs     : forwarded to base_optimizer constructor (lr, weight_decay, …)
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        assert rho >= 0.0, "rho must be non-negative"
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Perturb weights toward the sharpest direction (ascent step)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p.device)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w   # store perturbation for second step

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Restore weights and apply the base optimiser step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])   # undo perturbation

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError(
            "SAM requires two explicit steps. "
            "Call optimizer.first_step() then optimizer.second_step()."
        )

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2)

    # Delegate state_dict / load_state_dict to base optimiser so checkpoints work
    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
