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

"""
agent_integration.py - Google Agent Development Kit (ADK) Icin HFP Arayuzu

Bu dosya, HFP modelini bir "arac (tool)" veya arka plan servisi olarak
Google ADK (veya baska bir AI agent framework'u) ile kullanmak icin temel 
arayuzleri saglar.
"""

import torch
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

# Google ADK (veya baska Agent tool decorator'leri) projeye eklendiginde
# bu fonksiyonlari dogrudan tool olarak disari aktarabilirsiniz.
# Ornek kullanim: @agent_tool(description="HFP modeli ile metin/token tamamlama")

class HFPAgentWrapper:
    def __init__(self, config=None, device="cpu"):
        """
        Agent icin HFP modelini baslatir.
        
        Args:
            config (HFPConfig, optional): Model ayarlari. None ise varsayilan atanir.
            device (str): Cihaz ici calistirma hedefi (cpu, cuda vs.).
        """
        self.device = device
        if config is None:
            config = HFPConfig(
                vocab_size=200, 
                d_model=64, 
                n_layer=2, 
                n_head=4, 
                local_window=32,
                decay_mode="cubic_flux_chunked",
                key_feature_map="dpfp"
            )
        self.model = HFPForCausalLM(config).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_response(self, input_tokens, max_new_tokens=10):
        """
        Agent'in cagirdigi ana fonksiyon. HFP modeline diziyi verir ve 
        uretimi simule eder.
        
        Args:
            input_tokens (list of int): Girdi dizisi (token id listesi)
            max_new_tokens (int): Uretilecek maksimum token sayisi
            
        Returns:
            list of int: Uretilen yeni tokenlar
        """
        # HFP sabit bellek chunked calistirmayi destekler. 
        # Cihaz ici (on-device) ortamlarda bu cok kritiktir.
        x = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
        generated = []
        
        # Basit acgozlu (greedy) uretim dongusu
        for _ in range(max_new_tokens):
            outputs = self.model(x)
            next_token_logits = outputs.logits[0, -1]
            next_token = next_token_logits.argmax().item()
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]], device=self.device)], dim=1)
            
        return generated

# Bu dosya, ADK kurulumu tamamlandiginda @tool ve @agent decorator'leri 
# ile genisletilecektir.
