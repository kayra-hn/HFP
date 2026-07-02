import torch
import math

class AdamW_Thermodynamic(torch.optim.AdamW):
    """
    Thermodynamic Damping hook built on top of AdamW.
    Instead of freezing the network (like Vanilla SGD with exponential decay),
    this uses Momentum and Adam variance tracking to train large 200B models.
    It dampens the learning rate smoothly based on the Boltzmann Distribution
    but clamps it to a safe lower bound (e.g. 0.1x) to prevent Gradient Vanishing.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False, h_bar=1e-3, base_temp=1.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            group['h_bar'] = h_bar
            group['base_temp'] = base_temp
            group['original_lr'] = lr

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Calculate global gradient energy (L2 norm) across all parameters
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

            # Dampen learning rate instead of freezing gradients
            # clamp damping factor to not go below 0.1 to avoid freezing
            raw_damping = math.exp(-grad_energy * h_bar / (temp + 1e-8))
            damping_factor = max(raw_damping, 0.1) 
            group['lr'] = orig_lr * damping_factor

        # Execute standard AdamW step with the dampened learning rate
        super().step()

        return loss

class StiffTransientScheduler:
    """
    Scheduler that adjusts the 'base_temp' of the AdamW_Thermodynamic
    based on the rate of change of the loss (stiff equations).
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
        
        # During warmup, we don't cool down the manifold to allow initial projection
        if self.current_step < self.warmup_steps:
            return

        if current_loss is not None and self.last_loss is not None:
            # Calculate stiffness (rate of change of loss)
            stiffness = abs(current_loss - self.last_loss)
            
            # If stiffness is high (loss is jumping), we decrease the temperature 
            # to increase the damping factor in the optimizer.
            if stiffness > 1.0:
                for param_group in self.optimizer.param_groups:
                    param_group['base_temp'] *= self.cool_down_factor
                    
        self.last_loss = current_loss
