# 📡 Pipeline LLM pour les spécifications 3GPP

Projet de TP réalisé à l'ESMT Dakar : construire pas à pas un système de questions-réponses sur les normes télécoms 3GPP, du simple LLM jusqu'à une architecture multi-agents, et mesurer ce que chaque technique apporte.

## 🎯 Problème métier
Les ingénieurs télécoms passent beaucoup de temps à chercher des informations dans les spécifications 3GPP (documents de plusieurs centaines de pages). L'objectif est de tester si un petit LLM, aidé par la recherche documentaire (RAG) et le fine-tuning, peut répondre à des questions techniques sur ces normes.

## 📊 Données
- **TeleQnA** (GSMA, via Hugging Face) : questions sur les normes 3GPP, Releases 15 à 18 (LTE, 5G NR, cœur de réseau, sécurité, services).
- Fichiers utilisés : `TeleQnA_training.txt` (entraînement et corpus) et `TeleQnA_testing.txt` (évaluation).

## 🏗️ Les 7 phases

| Phase | Approche | Dossier |
|-------|----------|---------|
| 1 | Comparaison de 3 LLM (DistilGPT2, GPT2-Small, GPT2-Medium) | `Phase1/` |
| 2 | LLM + RAG (recherche dense FAISS) | `phase2/` |
| 3 | RAG avancé (BM25 + FAISS + reranking cross-encoder) | `Phase3/` |
| 4 | Fine-tuning QLoRA (PEFT, TRL) | `phase4/` |
| 5 | RAFT (RAG + fine-tuning) | `phase5/` |
| 6 | Agent RAG (raisonnement multi-étapes) | `phase6/` |
| 7 | Multi-agents + RAFT | `phase7/` |

## 📈 Résultats

### Protocole
- Chaque approche est évaluée sur **10 questions** du jeu de test (30 réponses en phase 1 : 10 par modèle).
- Métriques : ROUGE-1, ROUGE-2 et ROUGE-L entre la réponse générée et la réponse de référence.
- Le fine-tuning (phases 4 et 5) est un essai court : 1 époque, quelques pas d'entraînement.

Ces résultats sont donc **exploratoires** : ils permettent de comparer les approches entre elles dans les mêmes conditions, pas de conclure sur leurs performances réelles.

### Phase 1 : choix du modèle
| Modèle | ROUGE-L | Accuracy QCM (v2) |
|--------|---------|-------------------|
| DistilGPT2 | 0.245 | 1/10 |
| GPT2-Small | 0.210 | 1/10 |
| GPT2-Medium | 0.223 | 4/10 |

La première évaluation (ROUGE-L) a désigné **DistilGPT2**, qui est le modèle utilisé dans toutes les phases 2 à 7 (`pipeline_config.json`).

Après coup, j'ai refait l'évaluation sur les choix multiples (QCM), plus adaptée à TeleQnA (`Phase1_Comparaison_LLMs_3GPP_v2.ipynb`). GPT2-Medium y répond juste à 4 questions sur 10, contre 1 sur 10 pour les deux autres (`pipeline_config_v2.json`). Les phases 2 à 7 n'ont pas encore été relancées avec GPT2-Medium.

⚠️ Les scores ROUGE de la phase 1 viennent d'un protocole de génération différent de celui des phases 2 à 7 : ils ne sont **pas comparables** avec le tableau ci-dessous.

### Phases 2 à 7 : comparaison finale (DistilGPT2, même protocole, mêmes 10 questions)
| Approche | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| LLM seul | 0.050 | 0.010 | 0.043 |
| RAG | 0.056 | 0.007 | 0.055 |
| Fine-tuning QLoRA | 0.041 | 0.010 | 0.034 |
| **RAFT (RAG + fine-tuning)** 🏆 | **0.059** | 0.007 | **0.058** |
| Agent RAG | 0.040 | 0.005 | 0.040 |
| Multi-agents + RAFT | 0.041 | 0.005 | 0.041 |

Source : `phase7/phase7_comparaison_finale.csv`.

## 🔑 Ce que j'en retiens
- **RAFT obtient le meilleur ROUGE-L** : 0.058 contre 0.043 pour le LLM seul, soit **+37 % en relatif** sur ces 10 questions.
- **Le RAG seul apporte déjà l'essentiel du gain** (0.055). Le fine-tuning seul, trop court, fait moins bien que le modèle de base.
- **Les architectures agentiques n'ont pas amélioré les scores** avec un modèle de cette taille : elles ajoutent des étapes et du temps de calcul sans gain mesurable ici.
- **Les scores absolus restent bas** : GPT-2 est un petit modèle généraliste, et ROUGE pénalise les réponses correctes formulées différemment de la référence.

## 🚀 Pistes d'amélioration
- Évaluer sur un échantillon beaucoup plus large (plusieurs centaines de questions).
- Relancer les phases 2 à 7 avec GPT2-Medium, meilleur modèle selon l'évaluation QCM.
- Utiliser un LLM plus récent et plus grand, adapté aux questions techniques.
- Mesurer l'exactitude des réponses QCM pour toutes les phases, en plus de ROUGE.
- Remplacer les chemins Windows en dur par des chemins relatifs pour rendre les notebooks reproductibles.

## 🛠️ Technologies
- **LLM** : DistilGPT2, GPT-2 (Hugging Face Transformers)
- **Fine-tuning** : QLoRA avec PEFT et TRL
- **Recherche vectorielle** : FAISS, embeddings Sentence-Transformers (all-MiniLM-L6-v2)
- **Recherche lexicale** : BM25 (rank-bm25)
- **Reranking** : cross-encoder ms-marco-MiniLM
- **Évaluation** : ROUGE-1, ROUGE-2, ROUGE-L

## ⚙️ Installation
```bash
conda create -n tp_3gpp python=3.11
conda activate tp_3gpp
pip install transformers torch peft trl faiss-cpu
pip install sentence-transformers rank-bm25 evaluate
```
Télécharger ensuite TeleQnA depuis Hugging Face et adapter les chemins des fichiers au début de chaque notebook.

## 👩‍💻 Auteure
Marie Louise Hélène Gnilane Faye, élève-ingénieure IA & Data à l'ESMT Dakar
[LinkedIn](https://www.linkedin.com/in/gnilanefaye) · [GitHub](https://github.com/gnilanefaye)
