import sys
sys.stdout.reconfigure(encoding='utf-8')
from transformers import GPT2Tokenizer
import torch

def main():
    print("GPT2Tokenizer yükleniyor...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # first_words.py'ın ürettiği token ID'leri
    token_ids = [1, 15, 23, 48, 29640, 14008, 29370, 17978, 49827, 13148, 
                 6242, 48796, 12903, 22603, 49615, 5458, 10655, 25252, 9864, 47875, 
                 14542, 4168, 4664, 33629, 38330, 13016, 31758, 9214, 22141, 28962, 
                 28342, 40333, 21357, 45880]
    
    print("\nToken ID'ler:")
    print(token_ids)
    
    print("\nDecode Ediliyor...\n")
    print("=" * 50)
    decoded_text = tokenizer.decode(token_ids)
    print(decoded_text)
    print("=" * 50)
    
if __name__ == "__main__":
    main()
