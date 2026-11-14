import os
import sys
import uuid
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
    print(f"Embedding Model: IndoBERT ({INDOBERT_MODEL_NAME})")
    print("=" * 60)

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
            continue

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
