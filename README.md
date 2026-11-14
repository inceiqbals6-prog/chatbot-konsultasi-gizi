# Chatbot Konsultasi Gizi — Evaluasi Retrieval-Augmented Generation (RAG)

Repositori resmi kode sumber, modul arsitektur, dan artefak data evaluasi untuk Tugas Akhir / Skripsi:
> **"Analisis dan Perbandingan Kinerja Modul Retrieval serta Kualitas Generasi Respons pada Arsitektur Retrieval-Augmented Generation (RAG) untuk Chatbot Konsultasi Gizi Medis"**

---

## 🎯 Tujuan Penelitian

Penelitian ini berfokus pada dua tujuan utama:
1. **Analisis Kinerja Modul Retrieval (MRR):**  
   Menganalisis dan membandingkan kinerja modul retrieval pada 4 konfigurasi: **Naive RAG**, **Pre-Retrieval Only**, **Post-Retrieval Only**, dan **Full Advanced RAG** menggunakan metrik *Mean Reciprocal Rank* (MRR), serta mengidentifikasi kontribusi tahapan *pre-retrieval* dan *post-retrieval* secara terpisah maupun kombinasinya terhadap penemuan dokumen target klinis.
2. **Analisis Kualitas Generasi Respons (Framework RAGAS):**  
   Menganalisis dan membandingkan kualitas respons pada 5 konfigurasi: **LLM Baseline**, **Naive RAG**, **Pre-Retrieval Only**, **Post-Retrieval Only**, dan **Full Advanced RAG** menggunakan framework RAGAS berdasarkan metrik ***Context Relevance***, ***Faithfulness***, dan ***Answer Relevance***, dengan evaluator *LLM Judge* (GPT-4o) terhadap 75 respons yang dihasilkan dari 15 *clinical vignettes*.

---

## 🏗️ Arsitektur Sistem (Ablasi RAG)

Arsitektur chatbot gizi medis dirancang secara modular untuk memungkinkan perbandingan komparatif antarkonfigurasi:

```
[ Pertanyaan Pengguna (Clinical Vignette) ]
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
[ LLM Baseline ]         [ Pre-Retrieval Optimization ]
(Tanpa Retrieval)         • Clinical Query Rewriting (GPT-4o-mini)
                                  │
                                  ▼
                         [ Hybrid Retrieval ]
                          • Vector Semantic Search (text-embedding-3-small)
                          • BM25 / Text Keyword Search
                          • Reciprocal Rank Fusion (RRF)
                                  │
                                  ▼
                         [ Post-Retrieval Optimization ]
                          • BGE Cross-Encoder Reranker (bge-reranker-v2-m3)
                                  │
                                  ▼
                         [ Generation Stage ]
                          • Grounded Response Generation (Gemini 2.5 Flash)
```

---

## 📊 Ringkasan Hasil Evaluasi

### 1. Evaluasi Kinerja Retrieval (15 Clinical Vignettes)
| Konfigurasi Retrieval | MRR | Rank 1 | Rank 2 | Rank 3 | Rank 4-5 | Not Found (RR=0) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive RAG** | **0,2155** | 2 | 2 | 1 | 0 | 10 |
| **Pre-Retrieval Only** | **0,3555** | 4 | 1 | 3 | 0 | 7 |
| **Post-Retrieval Only** | **0,5000** | 7 | 1 | 0 | 0 | 7 |
| **Full Advanced RAG** | **0,8555** | **12** | 1 | 1 | 0 | 1 |

### 2. Evaluasi Kualitas Respons Chatbot (RAGAS Benchmark - 75 Respons)
| Konfigurasi Sistem | Context Relevance | Faithfulness | Answer Relevance | Rata-rata Skor RAGAS |
| :--- | :---: | :---: | :---: | :---: |
| **LLM Baseline** | *N/A (0)\** | 0,4839 | 0,6505 | 0,3781 |
| **Naive RAG** | 0,3667 | 0,6748 | **0,6562** | 0,5659 |
| **Pre-Retrieval Only** | 0,6333 | 0,7980 | 0,6551 | 0,6955 |
| **Post-Retrieval Only** | 0,6667 | 0,7671 | 0,6407 | 0,6915 |
| **Full Advanced RAG** | **0,8333** | **0,8562** | 0,6441 | **0,7779** |

*\*Catatan: Context relevance pada Baseline dicatat 0 karena tidak melibatkan retrieval konteks.*

---

## 📁 Struktur Berkas Repositori

```text
├── README.md                                    <- Dokumentasi resmi proyek Tugas Akhir
├── CHATBOT_MODIFICATIONS_FOR_EVALUATION.md       <- Log modifikasi sistem untuk pengujian
├── CHATBOT_RAG_FLOW_DETAILED.md                 <- Alur detail pipeline RAG end-to-end
├── CHATBOT_SPECIFICATION.md                     <- Spesifikasi fungsional chatbot
├── CHATBOT_SYSTEM_PROCESS.md                    <- Diagram dan mekanisme pemrosesan pesan
├── RAGAS_EVALUATION_IMPLEMENTATION.md           <- Metodologi & implementasi framework RAGAS
│
└── Mobile/chefgenius_mobile/
    ├── scripts/
    │   ├── evaluasi_ragas_semua.py              <- Skrip otomatisasi pengujian RAGAS (GPT-4o)
    │   ├── evaluasi ragas latest.md             <- Laporan lengkap hasil evaluasi per skenario
    │   ├── evaluasi ragas latest.csv            <- Data tabular hasil evaluasi
    │   └── ragas_evaluation_results_v10.0.csv   <- Dataset resmi hasil pengujian v10.0
    │
    ├── supabase/
    │   ├── functions/gemini-proxy/              <- Edge Function backend RAG
    │   │   ├── rag/query_rewrite.ts             <- Modul Pre-Retrieval (Query Rewriting)
    │   │   ├── rag/retrieval.ts                 <- Modul Hybrid Search & Reciprocal Rank Fusion
    │   │   ├── rag/reranker.ts                  <- Modul Post-Retrieval (BGE Cross-Encoder)
    │   │   ├── rag/prompt.ts                    <- Clinical system prompt & guideline injection
    │   │   ├── vignettes_master_v10_data.ts     <- Dataset 15 clinical vignettes acuan
    │   │   └── utils/telemetry.ts               <- Pencatatan latensi & metadata retrieval
    │   └── migrations/                          <- Skema database, vektor pgvector, & telemetri
    │
    ├── lib/features/chatbot/                    <- Antarmuka (UI) & interaksi Chatbot Flutter
    └── assets/figures/                          <- Diagram & visualisasi grafik untuk naskah TA
```

---

## ⚙️ Cara Menjalankan Replikasi Evaluasi

### 1. Menjalankan Evaluasi RAGAS (Python)
Pastikan Python 3.10+ telah terpasang, lalu siapkan variabel lingkungan:
```bash
cd Mobile/chefgenius_mobile/scripts
pip install pandas numpy requests
export OPENAI_API_KEY="your-openai-api-key"
export SUPABASE_URL="your-supabase-url"
export SUPABASE_ANON_KEY="your-supabase-anon-key"
python evaluasi_ragas_semua.py
```

### 2. Menjalankan Supabase Edge Functions (Lokal / Deploy)
```bash
cd Mobile/chefgenius_mobile/supabase
supabase functions serve gemini-proxy --no-verify-jwt
```

---

## 📚 Dokumen Acuan Klinis (Knowledge Base)
Knowledge base disusun berdasarkan pedoman medis dan konsensus nasional:
1. **PNPK Tata Laksana Penyakit Ginjal Kronik** (Kemenkes RI)
2. **Konsensus Nutrisi pada Penyakit Ginjal Kronik** (PERNEFRI)
3. **Pedoman Pengelolaan dan Pencegahan Diabetes Melitus Tipe 2** (PERKENI)
4. **Panduan Tata Laksana Dislipidemia** (PERKENI)
5. **Pedoman Tata Laksana Hipertensi pada Penyakit Kardiovaskular** (PERKI)
