# Chatbot Konsultasi Gizi — Sistem Retrieval-Augmented Generation (RAG)

Sistem *Retrieval-Augmented Generation* (RAG) untuk layanan konsultasi gizi medis berbasis pedoman klinis nasional (Penyakit Ginjal Kronik, Diabetes Melitus Tipe 2, Dislipidemia, dan Obesitas).

---

## Tech Stack

| Komponen | Teknologi |
| :--- | :--- |
| **Bahasa Pemrograman** | Python 3.10+, TypeScript (Deno) |
| **Backend / Serverless** | Supabase Edge Functions |
| **Basis Data Vektor** | PostgreSQL dengan ekstensi `pgvector` |
| **Model LLM** | Google Gemini 3 Flash Preview (`gemini-3-flash-preview`) |
| **Model Embedding** | IndoBERT (`firqaaa/indo-sentence-bert-base`) |
| **Model Reranker** | BAAI/bge-reranker-v2-m3 (Cross-Encoder) |
| **Metode Retrieval** | Hybrid Search (Vector + Full-Text) & Reciprocal Rank Fusion (RRF) |
| **Framework Evaluasi** | RAGAS (Faithfulness, Context Relevance, Answer Relevance) |

---

## Struktur Direktori

```text
├── main.py              # Antarmuka CLI interaktif untuk mencoba chatbot (kueri bebas)
├── evaluasi.py          # Skrip otomatisasi pengujian eksperimen RAGAS
├── requirements.txt     # Dependensi Python
├── README.md            # Dokumentasi sistem
│
├── data/
│   ├── vignettes_master_v10.json         # Dataset 15 skenario clinical vignettes acuan
│   ├── hasil_evaluasi_ragas.csv          # Hasil detail pengujian RAGAS
│   ├── ringkasan_evaluasi.md             # Rekapitulasi hasil evaluasi
│   └── ragas_evaluation_results_v10.0.csv# Dataset mentah evaluasi v10.0
│
└── supabase/
    ├── functions/gemini-proxy/           # Modul backend pipeline RAG (Edge Function)
    │   ├── rag/query_rewrite.ts          # Tahap pre-retrieval (Query Rewriting)
    │   ├── rag/retrieval.ts              # Hybrid Retrieval & Reciprocal Rank Fusion
    │   ├── rag/reranker.ts               # Tahap post-retrieval (BGE Cross-Encoder Reranker)
    │   ├── rag/prompt.ts                 # Konstruksi prompt klinis
    │   └── utils/telemetry.ts            # Pencatatan telemetri latensi & peringkat
    └── migrations/                       # Skema basis data (tabel dokumen, vektor, & telemetri)
```

---

## Panduan Menjalankan

### 1. Instalasi Dependensi
Pastikan Python 3.10+ telah terpasang di komputer Anda. Pasang pustaka yang diperlukan:
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Chatbot Interaktif (Kueri Bebas)
Untuk menguji chatbot langsung melalui terminal:
```bash
python main.py
```
Pada saat dijalankan, pengguna dapat:
- Memilih konfigurasi arsitektur RAG yang ingin diuji (Full Advanced RAG, Post-Retrieval Only, Pre-Retrieval Only, Naive RAG, atau LLM Baseline).
- Mengetikkan pertanyaan konsultasi gizi secara bebas (sistem secara default menggunakan embedding IndoBERT).
- Mengetik `mode` untuk mengganti konfigurasi RAG, atau `keluar` untuk selesai.

### 3. Menjalankan Skrip Evaluasi Eksperimen
Untuk menjalankan pengujian otomatis terhadap skenario klinis dan menghitung metrik evaluasi RAGAS:
```bash
python evaluasi.py
```
Hasil pengujian akan otomatis tersimpan dalam direktori `data/`.

---

## Basis Pengetahuan Klinis (Knowledge Base)

Basis pengetahuan medis disusun dari pedoman dan konsensus klinis resmi:
1. **PNPK Tata Laksana Penyakit Ginjal Kronik** (Kementerian Kesehatan RI)
2. **Konsensus Nutrisi pada Penyakit Ginjal Kronik** (PERNEFRI)
3. **Pedoman Pengelolaan dan Pencegahan Diabetes Melitus Tipe 2 Dewasa** (PERKENI)
4. **Panduan Tata Laksana Dislipidemia** (PERKENI)
5. **Pedoman Nasional Pelayanan Kedokteran Tata Laksana Obesitas Dewasa** (Kementerian Kesehatan RI)
