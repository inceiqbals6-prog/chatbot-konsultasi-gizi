# Chatbot Konsultasi Gizi - Sistem Retrieval-Augmented Generation (RAG)

Repositori ini berisi implementasi kode sumber modul *backend* RAG dan skrip evaluasi untuk penelitian Tugas Akhir:
**"Analisis dan Perbandingan Kinerja Modul Retrieval serta Kualitas Generasi Respons pada Arsitektur Retrieval-Augmented Generation (RAG) untuk Chatbot Konsultasi Gizi Medis"**

---

## 1. Ringkasan Penelitian

Penelitian ini membandingkan kinerja sistem pada beberapa konfigurasi ablasi RAG:
1. **Modul Retrieval (Evaluasi MRR):** Menguji kemampuan penemuan dan pemeringkatan dokumen acuan klinis pada 15 skenario *clinical vignettes* (gizi dan nefrologi) menggunakan metrik *Mean Reciprocal Rank* (MRR) pada 4 konfigurasi: Naive RAG, Pre-Retrieval Only, Post-Retrieval Only, dan Full Advanced RAG.
2. **Kualitas Generasi Respons (Evaluasi RAGAS):** Mengukur kualitas 75 respons pada 5 konfigurasi (LLM Baseline, Naive RAG, Pre-Retrieval Only, Post-Retrieval Only, dan Full Advanced RAG) menggunakan metrik *Context Relevance*, *Faithfulness*, dan *Answer Relevance* dengan evaluator *LLM Judge* (GPT-4o).

### Ringkasan Hasil Evaluasi

**Tabel 1. Perbandingan Kinerja Retrieval (MRR)**
| Konfigurasi Retrieval | MRR | Rank 1 | Rank 2 | Rank 3 | Rank 4-5 | Not Found (RR=0) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive RAG | 0,2155 | 2 | 2 | 1 | 0 | 10 |
| Pre-Retrieval Only | 0,3555 | 4 | 1 | 3 | 0 | 7 |
| Post-Retrieval Only | 0,5000 | 7 | 1 | 0 | 0 | 7 |
| Full Advanced RAG | **0,8555** | **12** | 1 | 1 | 0 | 1 |

**Tabel 2. Evaluasi Kualitas Respons Chatbot (RAGAS)**
| Konfigurasi Sistem | Context Relevance | Faithfulness | Answer Relevance | Rata-rata RAGAS |
| :--- | :---: | :---: | :---: | :---: |
| LLM Baseline | N/A (0)* | 0,4839 | 0,6505 | 0,3781 |
| Naive RAG | 0,3667 | 0,6748 | 0,6562 | 0,5659 |
| Pre-Retrieval Only | 0,6333 | 0,7980 | 0,6551 | 0,6955 |
| Post-Retrieval Only | 0,6667 | 0,7671 | 0,6407 | 0,6915 |
| Full Advanced RAG | **0,8333** | **0,8562** | 0,6441 | **0,7779** |

*\*Catatan: Nilai 0 pada Context Relevance LLM Baseline merupakan penanda ketiadaan konteks retrieval, bukan hasil penilaian kualitas konteks.*

---

## 2. Struktur Direktori Repositori

```text
├── main.py              # Program CLI untuk menguji chatbot interaktif dengan kueri bebas
├── evaluasi.py          # Skrip otomatisasi pengujian 15 skenario & penghitungan metrik RAGAS
├── requirements.txt     # Dependensi Python yang dibutuhkan
├── README.md            # Dokumentasi proyek
│
├── data/
│   ├── vignettes_master_v10.json         # Data 15 skenario clinical vignettes acuan
│   ├── hasil_evaluasi_ragas.csv          # Hasil detail 75 evaluasi RAGAS resmi
│   ├── ringkasan_evaluasi.md             # Laporan lengkap rekapitulasi evaluasi
│   └── ragas_evaluation_results_v10.0.csv# Dataset mentah evaluasi v10.0
│
└── supabase/
    ├── functions/gemini-proxy/           # Modul backend RAG (Edge Function)
    │   ├── rag/query_rewrite.ts          # Tahap pre-retrieval (Query Rewriting)
    │   ├── rag/retrieval.ts              # Hybrid Search & Reciprocal Rank Fusion (RRF)
    │   ├── rag/reranker.ts               # Tahap post-retrieval (BGE Cross-Encoder Reranker)
    │   ├── rag/prompt.ts                 # Template prompt & injeksi konteks medis
    │   └── utils/telemetry.ts            # Pencatatan telemetri latensi & peringkat chunk
    └── migrations/                       # Skema basis data (pgvector, tabel dokumen & telemetri)
```

---

## 3. Cara Menjalankan

### Persiapan Dependensi
Pastikan Python 3.10+ telah terpasang. Pasang dependensi yang diperlukan:
```bash
pip install -r requirements.txt
```

### A. Mencoba Chatbot Secara Interaktif (Kueri Bebas)
Untuk menguji chatbot dengan pertanyaan bebas buatan sendiri, jalankan:
```bash
python main.py
```
Pada program ini, pengguna dapat:
1. Memilih konfigurasi RAG yang ingin diuji (Full Advanced RAG, Post-Retrieval Only, Pre-Retrieval Only, Naive RAG, atau LLM Baseline).
2. Memilih model embedding (IndoBERT atau Gemini Embedding).
3. Mengajukan pertanyaan klinis secara langsung melalui terminal.
4. Mengetik `mode` untuk mengganti konfigurasi kapan saja, atau `keluar` untuk mengakhiri program.

### B. Menjalankan Uji Evaluasi Otomatis (Replikasi Penelitian)
Untuk mereplikasi atau menjalankan pengujian otomatis 15 skenario klinis ke 5 konfigurasi sistem:
```bash
python evaluasi.py
```
Skrip akan mengeksekusi pengujian, melakukan evaluasi *faithfulness*, *context relevance*, dan *answer relevance* menggunakan evaluator GPT-4o, serta memperbarui berkas rekapitulasi di dalam folder `data/`.

---

## 4. Basis Pengetahuan Klinis (Knowledge Base)
Dokumen acuan yang digunakan dalam basis data medis mencakup:
1. Pedoman Nasional Pelayanan Kedokteran (PNPK) Tata Laksana Penyakit Ginjal Kronik (Kemenkes RI)
2. Konsensus Nutrisi pada Penyakit Ginjal Kronik (PERNEFRI)
3. Pedoman Pengelolaan dan Pencegahan Diabetes Melitus Tipe 2 (PERKENI)
4. Panduan Tata Laksana Dislipidemia (PERKENI)
5. Pedoman Tata Laksana Obesitas Dewasa (Kemenkes RI)
