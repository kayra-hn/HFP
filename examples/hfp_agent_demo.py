# hfp_agent_demo.py
# Bu dosya HFP modelini Google ADK (veya baska bir Agent framework'u) ile 
# entegre etmenin temel bir ornegidir.

import os
from hfp.agent_integration import HFPAgentWrapper

# Eger google-adk kütüphanesinden gercek ajan nesneleri kullanacaksaniz
# importlari asagidaki gibi yapabilirsiniz (Framework surumune gore degisebilir):
# from google_adk.core import Agent, Tool

# 1. HFP modelini baslatiyoruz (CPU uzerinde, kucuk varsayilan ayarlar ile)
print("HFP modeli baslatiliyor (CPU)...")
hfp_model = HFPAgentWrapper(device="cpu")
print("HFP modeli basariyla yuklendi!")

# 2. HFP'yi bir Ajan "Araci (Tool)" haline getiren sarmalayici fonksiyon
def hfp_metin_uret(input_tokens: list) -> list:
    """
    HFP modelini kullanarak verilen token dizisinden yeni tokenlar uretir.
    (Gercek bir projede, input_text (str) alip tokenizer ile cevirmek daha dogrudur)
    """
    return hfp_model.generate_response(input_tokens, max_new_tokens=15)

# --- GOOGLE ADK TEMSILİ KULLANIMI ---

# Asagidaki kodlar Google ADK tam konfigure edildiginde (API key vs. eklendiginde)
# calisacak sekilde bir taslaktir:

"""
hfp_tool = Tool(
    name="HFP_Causal_LM",
    description="Özel O(1) bellekli HFP mimarisiyle token uretimi ve metin tamamlama yapar.",
    func=hfp_metin_uret
)

my_agent = Agent(
    name="HFP_Asistani",
    instructions=(
        "Sen HFP modelini test eden ve onunla metin uretimi saglayan bir asistansin. "
        "Kullanici senden metin uretmeni istediginde 'HFP_Causal_LM' aracini kullan "
        "ve donen sonuclari yorumla."
    ),
    tools=[hfp_tool]
)

# Test calistirmasi
if __name__ == "__main__":
    # yanit = my_agent.run("Deneme metni tokenlari...")
    # print(yanit)
    pass
"""

if __name__ == "__main__":
    # Tool'un yalın halini test edelim (Tokenizer olmadigi icin rastgele id'ler veriyoruz)
    ornek_tokenlar = [10, 15, 20, 25]
    print(f"\nOrnek Token Girdisi: {ornek_tokenlar}")
    uretilen = hfp_metin_uret(ornek_tokenlar)
    print(f"HFP Tarafindan Uretilen Tokenlar: {uretilen}")
    print("\nADK entegrasyon taslagi hazir. Modeli bir tool olarak kullanabilirsiniz.")
