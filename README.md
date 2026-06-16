# 📡 Pipeline LLM pour les Spécifications 3GPP

## 🎯 Problème métier
Les ingénieurs télécom passent des heures à chercher des informations
dans les spécifications 3GPP (documents de 1000+ pages).
Ce projet construit un système de QA automatique capable de répondre
à des questions techniques 3GPP en progressant du LLM simple
vers une architecture Multi-Agents avancée.

## 📊 Dataset
- **TeleQnA** (GSMA/ot-full via Hugging Face)
- 10 000 questions sur les normes 3GPP Release 15-18
- Domaines : LTE, 5G NR, Core Network, Security, Services

## 🏗️ Pipeline — 7 Phases

| Phase | Approche | ROUGE-L |
|-------|----------|---------|
| 1 | Comparaison 3 LLMs (DistilGPT2, GPT2-Small, GPT2-Medium) | 0.244 |
| 2 | LLM + RAG (FAISS Dense Retrieval) | 0.020 |
| 3 | RAG Avancé (BM25 + FAISS + Reranking) | 0.026 |
| 4 | Fine-Tuning QLoRA | 0.030 |
| 5 | RAFT (RAG + Fine-Tuning) 🏆 | 0.058 |
| 6 | Agent RAG (raisonnement multi-étapes) | 0.040 |
| 7 | Multi-Agents + RAFT | 0.041 |

## 🛠️ Technologies
- **LLMs** : DistilGPT2, GPT2 (Hugging Face Transformers)
- **Fine-Tuning** : QLoRA via PEFT + TRL
- **Vector DB** : FAISS (Facebook AI)
- **Embeddings** : Sentence-Transformers (all-MiniLM-L6-v2)
- **Sparse Search** : BM25 (rank-bm25)
- **Reranking** : Cross-Encoder (ms-marco-MiniLM)
- **Évaluation** : ROUGE-1, ROUGE-2, ROUGE-L
- **Versioning** : Git + GitHub

## 📁 Structure du projet
TP-LLM-3GPP-Pipeline/

├── Phase1/ → Comparaison LLMs

├── Phase2/ → LLM + RAG

├── Phase3/ → RAG Avancé

├── Phase4/ → Fine-Tuning QLoRA

├── Phase5/ → RAFT

├── Phase6/ → Agent RAG

├── Phase7/ → Multi-Agents + RAFT

└── pipeline_config.json

## 🔑 Résultat clé
**RAFT surpasse toutes les autres approches** avec +37% de ROUGE-L
vs le LLM de base, démontrant que la combinaison RAG + Fine-Tuning
est plus efficace que chaque technique séparément.

## ⚙️ Installation

```bash
conda create -n tp_3gpp python=3.11
conda activate tp_3gpp
pip install transformers torch peft trl faiss-cpu
pip install sentence-transformers rank-bm25 evaluate
```

## 👩‍💻 Auteure
Marie Louise Hélène Gnilane Faye