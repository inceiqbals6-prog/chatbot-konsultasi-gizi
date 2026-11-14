import os
import sys
import uuid
import requests

# Konfigurasi Default Sistem
DEFAULT_SUPABASE_URL = "https://zfiyfhmsuhitytsuioml.supabase.co"
DEFAULT_USER_EMAIL = "evaluator@example.com"

# Baca environment variable atau file .env jika ada
SUPABASE_URL = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL)
ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Jika belum di-set di environment, coba baca dari file konfigurasi lokal
def load_local_env():
    global ANON_KEY, SERVICE_ROLE_KEY, SUPABASE_URL
    possible_env_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "supabase", "functions", "gemini-proxy", "scripts", ".env"),
        os.path.join(os.path.dirname(__file__), "Mobile", "chefgenius_mobile", "dart_defines.env"),
        os.path.join(os.path.dirname(__file__), "Mobile", "chefgenius_mobile", "supabase", "functions", "gemini-proxy", "scripts", ".env"),
    ]
    for path in possible_env_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
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

# Mode RAG & Embedding
RAG_MODES = {
    "1": ("advanced", "Full Advanced RAG (Query Rewriting + Hybrid RRF + BGE Reranker)"),
    "2": ("post_retrieval", "Post-Retrieval Only (BGE Cross-Encoder Reranker)"),
    "3": ("pre_retrieval", "Pre-Retrieval Only (Query Rewriting + Hybrid RRF)"),
    "4": ("naive", "Naive RAG (Pencarian Vektor Standar)"),
    "5": ("none", "LLM Baseline (Tanpa Retrieval / Zero-shot)")
}

EMBEDDING_ENGINES = {
    "1": ("indobert", "IndoBERT (firqaaa/indo-sentence-bert-base) - Sesuai Evaluasi Resmi"),
    "2": ("gemini", "Gemini Embedding (gemini-embedding-001)")
}

def get_auth_token():
    """Mendapatkan session token terverifikasi untuk memanggil Supabase Edge Function."""
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
    print("\n" + "=" * 60)
    print("           PILIHAN KONFIGURASI ARSITEKTUR RAG")
    print("=" * 60)
    for k, v in RAG_MODES.items():
        prefix = "-> [1] (Default)" if k == "1" else f"   [{k}]"
        print(f"{prefix} : {v[1]}")
    
    choice_rag = input("\nPilih konfigurasi [1-5, Enter = 1]: ").strip()
    if choice_rag not in RAG_MODES:
        choice_rag = "1"
    selected_mode = RAG_MODES[choice_rag][0]
    selected_mode_label = RAG_MODES[choice_rag][1]

    print("\n" + "-" * 60)
    print("               PILIHAN MODEL EMBEDDING")
    print("-" * 60)
    for k, v in EMBEDDING_ENGINES.items():
        prefix = "-> [1] (Default)" if k == "1" else f"   [{k}]"
        print(f"{prefix} : {v[1]}")

    choice_emb = input("\nPilih model embedding [1-2, Enter = 1]: ").strip()
    if choice_emb not in EMBEDDING_ENGINES:
        choice_emb = "1"
    selected_emb = EMBEDDING_ENGINES[choice_emb][0]
    selected_emb_label = EMBEDDING_ENGINES[choice_emb][1]

    return selected_mode, selected_mode_label, selected_emb, selected_emb_label

def main():
    print("\n" + "=" * 65)
    print("       CHATBOT KONSULTASI GIZI MEDIS (CLI INTERAKTIF)")
    print("   Sistem Tugas Akhir: Retrieval-Augmented Generation (RAG)")
    print("=" * 65)

    mode, mode_label, emb, emb_label = choose_configuration()
    session_id = f"cli-{uuid.uuid4().hex[:8]}"

    print("\n[Menghubungkan ke layanan Supabase & LLM...]")
    token = get_auth_token()
    if not token:
        print("\n[PERINGATAN] Token otentikasi tidak ditemukan. Pastikan variabel lingkungan")
        print("SUPABASE_ANON_KEY dan SUPABASE_SERVICE_ROLE_KEY telah terpasang.")
        print("Mencoba melanjutkan dengan kunci publik...")
        token = ANON_KEY

    print("\n" + "=" * 65)
    print("Chatbot siap digunakan!")
    print(f"Konfigurasi aktif : {mode_label}")
    print(f"Embedding aktif   : {emb_label}")
    print("Ketik pertanyaan klinis Anda secara bebas di bawah.")
    print("Ketik 'mode' untuk mengganti pengaturan, atau 'keluar' untuk selesai.")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("\nPertanyaan Anda: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSelesai. Terima kasih!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["keluar", "exit", "quit", "q"]:
            print("\nSelesai. Terima kasih telah mencoba Chatbot Konsultasi Gizi!")
            break

        if user_input.lower() == "mode":
            mode, mode_label, emb, emb_label = choose_configuration()
            print(f"\n[OK] Pengaturan diperbarui ke: {mode_label} | {emb_label}\n")
            continue

        print("\n[Memproses pertanyaan...]")
        headers = {
            "apikey": ANON_KEY,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "action": "chatbot",
            "userMessage": user_input,
            "rag_mode": mode,
            "embedding_engine": emb,
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
                print("\n" + "-" * 65)
                print("JAWABAN CHATBOT:")
                print("-" * 65)
                print(answer)
                print("-" * 65)
            else:
                print(f"\n[Error {resp.status_code}] Gagal mendapatkan respons dari server.")
                print(resp.text[:300])
        except requests.exceptions.Timeout:
            print("\n[Error] Waktu permintaan habis (timeout). Silakan coba lagi.")
        except Exception as e:
            print(f"\n[Error] Terjadi kendala koneksi: {e}")

if __name__ == "__main__":
    main()
