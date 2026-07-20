import os
import sys
import random
import traceback
import google.generativeai as genai
from google.generativeai.types import content_types

try:
    from hfp.agent_integration import HFPAgentWrapper
except ImportError:
    print("HATA: 'hfp' paketi bulunamadı.")
    sys.exit(1)

print("HFP Tool başlatılıyor...")
hfp_model = HFPAgentWrapper(device="cpu")

# ------------------------------------------------------------
# ARAÇLAR (her parametre string, içeride güvenli dönüşüm)
# ------------------------------------------------------------
def hfp_tool_fonksiyonu(input_tokens_str: str) -> str:
    try:
        print(f"[TOOL HFP] Gelen arg: {input_tokens_str!r}")
        token_list = [int(float(x.strip())) for x in input_tokens_str.split(",") if x.strip()]
        print(f"[TOOL HFP] Token list: {token_list[:10]}...")
        uretilen = hfp_model.generate_response(token_list, max_new_tokens=15)
        return f"Üretilen: {uretilen}"
    except Exception:
        return f"[HFP HATASI]\n{traceback.format_exc()}"

def zorlu_mqar_testi_yap(baglam_uzunlugu: str, anahtar_sayisi: str, aldatmaca_orani: str) -> str:
    try:
        print(f"[TOOL MQAR] Argümanlar: L={baglam_uzunlugu!r}, K={anahtar_sayisi!r}, O={aldatmaca_orani!r}")
        L = int(float(baglam_uzunlugu))
        K = int(float(anahtar_sayisi))
        O = float(aldatmaca_orani)
        print(f"[TOOL MQAR] Dönüşüm: L={L} ({type(L)}), K={K} ({type(K)}), O={O} ({type(O)})")

        if L < (K * 2) + 10:
            return f"Hata: Bağlam çok kısa. En az {(K * 2) + 10} olmalı."

        dizi = [10, 99]
        tekrar = int(L - 4)   # gereksiz ama emin olalım
        if tekrar < 0:
            return "Hata: Bağlam 4'ten küçük."

        for i in range(tekrar):
            if random.random() < O:
                dizi.extend([10, random.randint(1, 98)])
            else:
                dizi.append(random.randint(1, 9))

        dizi.extend([10, 0])
        temiz_dizi = [int(x) for x in dizi]
        print(f"[TOOL MQAR] HFP'ye gönderiliyor... (ilk 20): {temiz_dizi[:20]}")

        uretilen = hfp_model.generate_response(temiz_dizi, max_new_tokens=1)
        if not uretilen:
            return "Hata: HFP boş cevap döndü."
        tahmin = int(uretilen[0])
        sonuc = "BAŞARILI" if tahmin == 99 else "BAŞARISIZ"
        return f"Sonuç: {sonuc} | Tahmin: {tahmin}, Beklenen: 99"

    except Exception:
        return f"[MQAR HATASI]\n{traceback.format_exc()}"

# ------------------------------------------------------------
# MANUEL FONKSİYON ÇAĞRI YÖNETİCİSİ
# ------------------------------------------------------------
available_tools = {
    "hfp_tool_fonksiyonu": hfp_tool_fonksiyonu,
    "zorlu_mqar_testi_yap": zorlu_mqar_testi_yap,
}

def handle_function_call(function_call):
    """Gemini'den gelen fonksiyon çağrısını işler."""
    name = function_call.name
    args = function_call.args
    print(f"\n[MANUEL] Çağrı: {name}, argümanlar: {args}")

    # Her argümanı güvenli hale getir: önce string yap, sayıysa dönüştürmeyi tool halleder
    clean_args = {}
    for k, v in args.items():
        # Gelen değeri direkt string olarak kullan (None, float vs. olabilir)
        clean_args[k] = str(v)
        print(f"[MANUEL]   {k}: {v!r} -> {clean_args[k]!r}")

    func = available_tools.get(name)
    if not func:
        return f"Hata: '{name}' bulunamadı."

    try:
        result = func(**clean_args)
        print(f"[MANUEL] Sonuç: {result}")
        return result
    except Exception as e:
        return f"Fonksiyon çağrısı hatası: {traceback.format_exc()}"

# ------------------------------------------------------------
# ANA UYGULAMA
# ------------------------------------------------------------
def ana_uygulama():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("HATA: GEMINI_API_KEY bulunamadı!"); sys.exit(1)

    genai.configure(api_key=api_key, transport='rest')

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        tools=[hfp_tool_fonksiyonu, zorlu_mqar_testi_yap],
        system_instruction=(
            "Sen bir Red Team ajanısın. Test için 'zorlu_mqar_testi_yap' aracını kullan. "
            "Tüm sayısal değerleri METİN olarak gönder (örnek: '2000')."
        )
    )

    chat = model.start_chat(enable_automatic_function_calling=False)
    print("\n✅ Manuel mod başladı. 'HFP'yi 2000 bağlam ile test et' yaz.\n")

    while True:
        try:
            user_in = input("Sen: ")
            if user_in.lower() in ["exit", "quit", "cikis"]:
                break

            response = chat.send_message(user_in)

            # Cevapta fonksiyon çağrısı var mı kontrol et
            while True:
                parts = response.candidates[0].content.parts
                function_calls = [p for p in parts if "function_call" in p]
                if not function_calls:
                    # Başka çağrı yok, son metni yazdır
                    if response.text:
                        print(f"Ajan: {response.text}")
                    break

                # Tüm fonksiyon çağrılarını işle
                function_responses = []
                for fc in function_calls:
                    result = handle_function_call(fc.function_call)
                    function_responses.append(
                        genai.protos.Part(function_response=genai.protos.FunctionResponse(
                            name=fc.function_call.name,
                            response={"result": result}
                        ))
                    )

                # Modelin bu cevapları işlemesi için tekrar gönder
                response = chat.send_message(function_responses)

        except KeyboardInterrupt:
            print("\nÇıkış yapıldı.")
            break
        except Exception:
            print(f"\n[AJAN HATASI]\n{traceback.format_exc()}")

if __name__ == "__main__":
    ana_uygulama()