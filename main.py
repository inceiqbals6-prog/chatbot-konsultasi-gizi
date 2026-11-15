import os
import sys
import uuid
import warnings
import logging
import threading

# Hilangkan log, peringatan unauthenticated HF Hub, dan progress bar transformers
threading.excepthook = lambda args: None
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    from transformers import logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass

import requests

DEFAULT_SUPABASE_URL = "https://zfiyfhmsuhitytsuioml.supabase.co"
DEFAULT_USER_EMAIL = "evaluator@example.com"
DEFAULT_EMBEDDING_ENGINE = "indobert"
INDOBERT_MODEL_NAME = "firqaaa/indo-sentence-bert-base"

SUPABASE_URL = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL)
ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

def load_local_env():
    global ANON_KEY, SERVICE_ROLE_KEY, SUPABASE_URL
    paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "supabase", "functions", "gemini-proxy", "scripts", ".env"),
        os.path.join(os.path.dirname(__file__), "Mobile", "chefgenius_mobile", "dart_defines.env"),
        os.path.join(os.path.dirname(__file__), "Mobile", "chefgenius_mobile", "supabase", "functions", "gemini-proxy", "scripts", ".env"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip('"').strip("'")
                            if k in ["SUPABASE_KEY", "SUPABASE_ANON_KEY"] and not ANON_KEY:
                                ANON_KEY = v
                            elif k == "SUPABASE_SERVICE_ROLE_KEY" and not SERVICE_ROLE_KEY:
                                SERVICE_ROLE_KEY = v
                            elif k == "SUPABASE_URL" and SUPABASE_URL == DEFAULT_SUPABASE_URL:
                                SUPABASE_URL = v
            except Exception:
                pass

load_local_env()

_indobert_tokenizer = None
_indobert_model = None
_indobert_available = None

def init_indobert():
    global _indobert_tokenizer, _indobert_model, _indobert_available
    if _indobert_available is not None:
        return _indobert_available
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        print("Memuat model IndoBERT (firqaaa/indo-sentence-bert-base)...", end=" ", flush=True)
        try:
            _indobert_tokenizer = AutoTokenizer.from_pretrained(INDOBERT_MODEL_NAME, local_files_only=True)
            _indobert_model = AutoModel.from_pretrained(INDOBERT_MODEL_NAME, local_files_only=True)
        except Exception:
            _indobert_tokenizer = AutoTokenizer.from_pretrained(INDOBERT_MODEL_NAME)
            _indobert_model = AutoModel.from_pretrained(INDOBERT_MODEL_NAME)

        _indobert_model.eval()
        _indobert_available = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Siap - {device.upper()}]")
    except Exception as e:
        _indobert_available = False
        print(f"[Offline fallback aktif: {e}]")
    return _indobert_available

def get_indobert_embedding(text: str):
    if not init_indobert():
        return None
    try:
        import torch
        import torch.nn.functional as F
        enc = _indobert_tokenizer([str(text)], padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            out = _indobert_model(**enc)
        mask = enc['attention_mask'].unsqueeze(-1).expand(out[0].size()).float()
        mean_pooled = torch.sum(out[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        emb = F.normalize(mean_pooled, p=2, dim=1)[0].tolist()
        return emb
    except Exception:
        return None

RAG_MODES = {
    "1": ("advanced", "Full Advanced RAG (Query Rewriting + Hybrid RRF + BGE Reranker)"),
    "2": ("post_retrieval", "Post-Retrieval Only (BGE Cross-Encoder Reranker)"),
    "3": ("pre_retrieval", "Pre-Retrieval Only (Query Rewriting + Hybrid RRF)"),
    "4": ("naive", "Naive RAG (Pencarian Vektor Standar)"),
    "5": ("none", "LLM Baseline (Tanpa Retrieval)")
}

def get_auth_token():
    if not SERVICE_ROLE_KEY or not ANON_KEY:
        return None
    try:
        admin_headers = {
            "apikey": SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
            "Content-Type": "application/json"
        }
        res = requests.post(
            f"{SUPABASE_URL}/auth/v1/admin/generate_link",
            headers=admin_headers,
            json={"type": "magiclink", "email": DEFAULT_USER_EMAIL},
            timeout=15
        )
        token_hash = res.json().get("hashed_token")
        if not token_hash:
            return None

        verify_res = requests.post(
            f"{SUPABASE_URL}/auth/v1/verify",
            headers={"apikey": ANON_KEY, "Content-Type": "application/json"},
            json={"type": "magiclink", "token_hash": token_hash},
            timeout=15
        )
        return verify_res.json().get("access_token")
    except Exception:
        return None

def choose_configuration():
    print("\nPilih Konfigurasi RAG:")
    for k, v in RAG_MODES.items():
        prefix = "[1] (Default)" if k == "1" else f"[{k}]"
        print(f"  {prefix} {v[1]}")
    
    choice = input("Pilihan [1-5, default=1]: ").strip()
    if choice not in RAG_MODES:
        choice = "1"
    return RAG_MODES[choice]

def main():
    print("=" * 60)
    print("Chatbot Konsultasi Gizi Medis (CLI)")
    print(f"Embedding Engine: IndoBERT ({INDOBERT_MODEL_NAME})")
    print("=" * 60)

    if not ANON_KEY:
        print("\n[PERINGATAN] Berkas .env atau SUPABASE_ANON_KEY belum dikonfigurasi.")
        print("Silakan buat berkas .env (contoh: salin dari .env.example) dan isi kredensial Supabase Anda.")
        print("Program dihentikan untuk mencegah akses tidak sah ke kuota API.")
        return

    mode, mode_label = choose_configuration()
    session_id = f"cli-{uuid.uuid4().hex[:8]}"

    token = get_auth_token()
    if not token:
        token = ANON_KEY

    print("\n" + "-" * 60)
    print(f"Konfigurasi aktif : {mode_label}")
    print(f"Model Embedding   : IndoBERT ({INDOBERT_MODEL_NAME})")
    print("Ketik pertanyaan klinis Anda, 'mode' untuk ganti konfigurasi, atau 'keluar' untuk selesai.")
    print("-" * 60)

    # Inisialisasi model IndoBERT di awal jika mode membutuhkan retrieval
    if mode != "none":
        init_indobert()

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSelesai.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["keluar", "exit", "quit", "q"]:
            print("Selesai.")
            break

        if user_input.lower() == "mode":
            mode, mode_label = choose_configuration()
            print(f"\nMode diperbarui: {mode_label}")
            if mode != "none":
                init_indobert()
            continue

        query_emb = None
        if mode != "none":
            query_emb = get_indobert_embedding(user_input)

        headers = {
            "apikey": ANON_KEY,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "action": "chatbot",
            "userMessage": user_input,
            "rag_mode": mode,
            "embedding_engine": DEFAULT_EMBEDDING_ENGINE,
            "session_id": session_id,
            "app_version": "cli-1.0.0"
        }
        if query_emb:
            payload["query_embedding"] = query_emb

        try:
            resp = requests.post(
                f"{SUPABASE_URL}/functions/v1/gemini-proxy",
                headers=headers,
                json=payload,
                timeout=60
            )
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("result", "Tidak ada respon teks.")
                print(f"\nChatbot:\n{answer}")
            else:
                print(f"\nError {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.Timeout:
            print("\nError: Request timeout. Coba tanyakan lagi.")
        except Exception as e:
            print(f"\nError koneksi: {e}")

if __name__ == "__main__":
    main()
