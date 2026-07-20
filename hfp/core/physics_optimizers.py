# Hyper Flux Projection (HFP) — O(1)-memory causal language model
# Copyright (C) 2026 Kayrahan Yılmaz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import math

class AdamW_Thermodynamic(torch.optim.AdamW):
    """
    Thermodynamic Damping hook on top of AdamW (OPSIYONEL / DENEYSEL).

    [C4 GUNCELLEME] Damping yasasi, revize makaledeki LINEER RELAKSASYONA
    (sigma' = -sigma(1+sigma)) uyumlu hale getirildi. Onceki surum kubik akis
    (dtheta/dtau = -eta*theta^3 -> 1/sqrt(1+...)) kullaniyordu; bu, cop atilan
    eski teoriydi. Yeni yasa kucuk sigma'da lineer (usel) sonumlenme verir.

    Not: Bu optimizer eğitimin DEFAULT'u DEGILDIR. Sağlıklı baseline için train.py
    standart AdamW + cosine warmup kullanır. Bu sınıf, fizik-ilhamlı gradyan-adaptif
    LR fikrini A/B test etmek isteyenler için opsiyonel bir knob'dur. h_bar artik
    etkili bir aralikta (default 0.05) - eski 1e-3 degeri damping'i pratikte ~1
    yapip mekanizmayi olu birakiyordu.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False, h_bar=0.05, base_temp=1.0, eta_tilde=1.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            group['h_bar'] = h_bar
            group['base_temp'] = base_temp
            group['eta_tilde'] = eta_tilde
            group['original_lr'] = lr

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Global gradyan enerjisi (L2 norm)
        total_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
        grad_energy = math.sqrt(total_norm)

        for group in self.param_groups:
            h_bar = group['h_bar']
            temp = group['base_temp']
            orig_lr = group['original_lr']
            eta_tilde = group['eta_tilde']

            # [C4] Lineer/logistik relaksasyon: sigma' = -sigma(1+sigma) ->
            # algebraic (power-law) sonum: damping = 1/(1 + eta*s + eta*s^2),
            # kucuk s'de ~1/(1+eta*s) (lineer), buyuk s'de daha guclu sonum.
            s = grad_energy * h_bar / (temp + 1e-8)
            raw_damping = 1.0 / (1.0 + eta_tilde * s + eta_tilde * s * s)
            damping_factor = max(raw_damping, 0.1)  # gradient vanishing'i onlemek icin alt sinir
            group['lr'] = orig_lr * damping_factor

        super().step()
        return loss

class StiffTransientScheduler:
    """
    Scheduler that adjusts 'base_temp' of AdamW_Thermodynamic based on loss stiffness.
    (Yalnizca AdamW_Thermodynamic ile kullanilir; standart AdamW icin cosine warmup onerilir.)
    """
    def __init__(self, optimizer, warmup_steps=1000, cool_down_factor=0.99):
        if not isinstance(optimizer, AdamW_Thermodynamic):
            raise TypeError("StiffTransientScheduler requires an AdamW_Thermodynamic optimizer")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cool_down_factor = cool_down_factor
        self.current_step = 0
        self.last_loss = None

    def step(self, current_loss=None):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            return
        if current_loss is not None and self.last_loss is not None:
            stiffness = abs(current_loss - self.last_loss)
            if stiffness > 1.0:
                for param_group in self.optimizer.param_groups:
                    param_group['base_temp'] *= self.cool_down_factor
        self.last_loss = current_loss
