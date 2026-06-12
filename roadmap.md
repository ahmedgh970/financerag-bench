# 🗺️ Roadmap 12 semaines — RAG & Agentic RAG sur FinanceBench
### De chercheur à AI Engineer : un projet portfolio production-grade

> **Objectif final** : un repo GitHub unique, riche et professionnel, qui démontre : RAG complet sur FinanceBench → pipeline d'évaluation rigoureuse → benchmark multi-LLM open source → patterns agentiques (LangGraph) → workflow multi-agent → déploiement avec monitoring. Le tout documenté avec un reporting digne d'un ingénieur en production.

> **Budget temps** : 12h/semaine. Règle d'or : **80% build / 20% lecture-vidéos**. Si tu dépasses 3h de "consommation de contenu" dans la semaine, reviens au code.

---

## 0. La stack (les technos les plus demandées en entreprise — France 2026)

| Couche | Techno choisie | Pourquoi celle-là |
|---|---|---|
| Langage | **Python 3.11+** (typage, `uv` ou `poetry`) | Standard absolu |
| API backend | **FastAPI** | #1 dans les offres AI Engineer |
| Abstraction LLM | **LiteLLM** | Une seule interface pour tous les providers (OpenAI-compatible) |
| Orchestration agents | **LangChain + LangGraph** | Les plus cités dans les offres FR (LangChain, LangGraph, MCP) |
| Vector DB | **Qdrant** (+ pgvector en bonus) | Très demandé, open source, facile en Docker |
| Recherche lexicale | **BM25** (rank-bm25 ou Qdrant hybrid) | Hybrid search = attendu en entretien |
| Reranking | **Cross-encoder** (BAAI/bge-reranker, sentence-transformers) | Standard industrie |
| Parsing PDF | **Docling** (IBM) ou **Unstructured** | Le nerf de la guerre sur FinanceBench (tableaux !) |
| Embeddings | **BGE-M3** ou **multilingual-e5** (open source) + OpenAI text-embedding-3 en comparaison | Benchmarker les deux = bon réflexe |
| Évaluation | **Ragas** + métriques retrieval custom (recall@k, MRR, nDCG) | Ragas = la lib d'éval la plus citée |
| Observabilité | **Langfuse** (self-host ou cloud free tier) | LLMOps très demandé en France |
| Serving local | **Ollama** (dev) puis **vLLM** (benchmark sérieux) | vLLM = ton atout inference |
| LLMs open source | **Qwen 2.5/3 (7-14B)**, **Llama 3.1/3.3 8B-70B**, **Mistral Small**, **Gemma** | Voir semaine 6 |
| CI/CD | **GitHub Actions** | Standard |
| Conteneurisation | **Docker + docker-compose** | Standard |
| Config | **pydantic-settings + YAML** (ou Hydra) | Expériences reproductibles |
| UI démo | **Gradio** ou **Streamlit** | Déploiement HF Spaces |
| Reporting | **README riche + tableaux markdown + un dashboard Streamlit "Benchmark Explorer"** | Ton différenciateur |

**Accès LLM open source sans GPU coûteux :**
- **Groq** (https://console.groq.com) : Llama 3.3 70B, gratuit et ultra-rapide — parfait pour itérer.
- **Mistral La Plateforme** (https://console.mistral.ai) : free tier, modèles Mistral (bonus : entreprise française, bon signal).
- **Ollama** (https://ollama.com) : Qwen/Llama/Gemma en local sur ton laptop (versions 7-8B quantisées).
- **HF Inference Providers** (https://huggingface.co/docs/inference-providers) : accès à de nombreux modèles open weights.
- **vLLM** (https://docs.vllm.ai) : pour ton benchmark serving en fin de projet (GPU à louer ponctuellement : Colab Pro, Lightning AI, RunPod, ~10-20€ au total).

---

## 1. Architecture cible du repo

```
financerag-bench/                  # trouve un meilleur nom 😉
├── README.md                      # vitrine : archi, résultats, démo GIF
├── docs/                          # decisions (ADR), rapports de benchmark
├── configs/                       # 1 YAML = 1 expérience reproductible
├── data/
│   ├── pdfs/                      # 10-K/10-Q FinanceBench (script de download)
│   ├── golden_set/                # 150 QA FinanceBench + extensions
│   └── processed/                 # chunks, index
├── src/
│   ├── ingestion/                 # parsing PDF, chunking (3 stratégies)
│   ├── retrieval/                 # dense, BM25, hybrid, reranker
│   ├── llm/                       # client LiteLLM + prompts versionnés
│   ├── rag/                       # pipelines : naive → advanced → agentic
│   ├── agents/                    # LangGraph : router, grader, rewriter, multi-agent
│   ├── evaluation/                # métriques retrieval + Ragas + runner
│   └── api/                       # FastAPI
├── benchmarks/                    # scripts + résultats (JSON/CSV) versionnés
├── dashboard/                     # Streamlit benchmark explorer
├── tests/                         # pytest (unit + integration + eval regression)
├── .github/workflows/             # CI : lint, tests, smoke eval
├── docker-compose.yml             # api + qdrant + langfuse
└── Makefile                       # make ingest / eval / benchmark / serve
```

**Best practices transverses (à suivre dès la semaine 1) :**
- Commits atomiques + messages clairs (Conventional Commits : https://www.conventionalcommits.org).
- Branches + PR vers `main` même seul (tu te relis = revue de code).
- `ruff` (lint+format) + `mypy` léger + `pre-commit` (https://pre-commit.com).
- Chaque décision d'architecture → un mini **ADR** (Architecture Decision Record) dans `docs/` : 5 lignes (contexte, options, choix, pourquoi). Les recruteurs adorent. Format : https://adr.github.io
- Jamais de clé API dans le code → `.env` + pydantic-settings.
- Chaque résultat de benchmark = fichier JSON horodaté + config associée. **Si tu ne peux pas rejouer une expérience avec une commande, elle n'existe pas.**

---

# 📅 MOIS 1 — Fondations : données, éval, RAG baseline

> Mantra du mois : **"Eval first."** On ne touche pas aux optimisations tant que le harnais de mesure n'existe pas.

## Semaine 1 — Setup, FinanceBench, parsing PDF

**Objectif** : repo propre, données récupérées, premier pipeline d'ingestion qui tourne.

**Checklist :**
- [ ] Créer le repo GitHub (public), structure ci-dessus, README squelette avec le pitch du projet
- [ ] Setup : `uv`/poetry, ruff, pre-commit, pytest, Makefile, .env
- [ ] Télécharger FinanceBench : les 150 QA (JSONL) + les PDF associés (~50-80 docs)
- [ ] Explorer le dataset dans un notebook : types de questions (extraction de métrique, calcul, raisonnement), distribution par entreprise/document — note tes observations dans `docs/dataset_analysis.md`
- [ ] Module `ingestion/` : parser 3-5 PDF avec **Docling** ET **pymupdf**, comparer visuellement la qualité (surtout les tableaux !)
- [ ] Premier test pytest (ex : le parser retourne du texte non vide pour chaque PDF)

**Validation de fin de semaine** : `make ingest` parse tous les PDF et stocke les documents structurés. Tu sais dire quel parser gère le mieux les tableaux financiers et pourquoi (→ ADR n°1).

**Ressources :**
- 📄 Dataset : https://github.com/patronus-ai/financebench + https://huggingface.co/datasets/PatronusAI/financebench
- 📄 Paper FinanceBench (lecture rapide, 30 min) : https://arxiv.org/abs/2311.11944 — note le résultat clé : les RAG naïfs échouent ~80% du temps. C'est TA baseline à battre.
- 🔧 Docling : https://github.com/docling-project/docling | Unstructured : https://docs.unstructured.io
- 📝 Blog : "What we learned from parsing PDFs" — cherche les comparatifs Docling vs Unstructured vs LlamaParse sur le blog de chacun
- 🎥 Vidéo : "Docling: Document Parsing for RAG" (IBM, YouTube, ~15 min)

## Semaine 2 — Golden set & harnais d'évaluation retrieval

**Objectif** : le système de mesure AVANT le système. C'est la semaine qui te différencie de 95% des candidats.

**Checklist :**
- [ ] Module `evaluation/` : charger les 150 QA, normaliser en schéma pydantic (question, réponse gold, evidence, doc source, page)
- [ ] Implémenter les métriques retrieval : **recall@k, precision@k, MRR, nDCG** (à la main, ~100 lignes — tu sais faire, c'est de la recherche 😄). Le matching evidence↔chunk se fait par overlap de texte ou par page
- [ ] Implémenter le runner d'éval : prend une config YAML, exécute, sauvegarde un JSON de résultats dans `benchmarks/`
- [ ] Setup **Ragas** pour l'éval génération (faithfulness, answer relevancy, context precision/recall) — teste-le sur 5 questions à la main pour comprendre ce que mesure chaque métrique
- [ ] Décider ta métrique de réponse finale : les réponses FinanceBench sont souvent numériques → écrire un **juge LLM** simple (prompt : "la réponse candidate est-elle équivalente à la réponse gold ? oui/non + justification") + normalisation des nombres. Valide ton juge sur 20 cas à la main
- [ ] Test pytest : les métriques retrieval donnent les bons scores sur un mini cas synthétique

**Validation** : `make eval CONFIG=configs/dummy.yaml` produit un rapport JSON avec toutes les métriques sur un retrieval factice. Ton juge LLM est validé manuellement (accord > 90% avec ton jugement humain sur 20 cas).

**Ressources :**
- 📚 Ragas docs : https://docs.ragas.io (lis "Core Concepts" + "Metrics")
- 📝 Blog INCONTOURNABLE : Hamel Husain, "Your AI Product Needs Evals" : https://hamel.dev/blog/posts/evals/ — la bible de l'éval LLM côté industrie
- 📝 Eugene Yan, "Patterns for Building LLM-based Systems" : https://eugeneyan.com/writing/llm-patterns/ (section evals)
- 📄 Paper : "Lost in the Middle" https://arxiv.org/abs/2307.03172 (pourquoi la position du contexte compte — utile pour tes analyses)
- 🎥 "LLM Evaluation" — talks de Hamel Husain / Shreya Shankar sur YouTube

## Semaine 3 — RAG baseline (naïf) + premiers chiffres

**Objectif** : le pipeline le plus simple possible, mesuré de bout en bout. Résiste à la tentation d'optimiser !

**Checklist :**
- [ ] Qdrant en Docker (`docker-compose.yml`) : https://qdrant.tech/documentation/quickstart/
- [ ] Chunking fixe simple (ex : 512 tokens, overlap 64) → embeddings **BGE-M3** (local, gratuit) → index Qdrant
- [ ] Module `llm/` avec **LiteLLM** : interface unique, premier modèle branché = Llama 3.3 70B via **Groq** (gratuit)
- [ ] Pipeline `rag/naive.py` : question → top-k dense → prompt simple → réponse
- [ ] **Lancer l'éval complète sur les 150 questions** : retrieval (recall@k...) + génération (juge LLM + Ragas sur un sous-ensemble de 50 pour limiter les coûts)
- [ ] Documenter les résultats : tableau dans `docs/benchmark_report.md` + 5 exemples d'échecs analysés à la main (c'est l'or pour tes entretiens)

**Validation** : tu as TES premiers chiffres baseline (ex : recall@5 = X%, accuracy juge = Y%). Tu peux citer 3 modes d'échec typiques (tableau mal parsé, mauvaise page retrouvée, hallucination de chiffre).

**Ressources :**
- 📚 LiteLLM : https://docs.litellm.ai/docs/
- 📚 Qdrant quickstart + hybrid queries : https://qdrant.tech/documentation/
- 📄 Paper fondateur RAG (Lewis et al. 2020), lecture en diagonale : https://arxiv.org/abs/2005.11401
- 🎥 freeCodeCamp "RAG from Scratch" (playlist LangChain, Lance Martin) : https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x — excellent pour structurer ta tête

## Semaine 4 — Reporting v1 + qualité du repo + démo minimale

**Objectif** : transformer le travail en signal visible. Fin du mois 1 = quelque chose à montrer.

**Checklist :**
- [ ] CI GitHub Actions : lint (ruff) + tests pytest sur chaque push — badge dans le README
- [ ] README v1 sérieux : schéma d'architecture (draw.io / excalidraw), tableau de résultats baseline, instructions repro en 3 commandes
- [ ] Mini API FastAPI : endpoint `POST /ask` (question → réponse + sources citées + latence)
- [ ] UI Gradio basique branchée sur l'API, screenshot/GIF dans le README
- [ ] Rétrospective : 1 page `docs/month1_retro.md` (ce qui a marché, surpris, bloqué)
- [ ] 🎯 Bonus : premier post LinkedIn ("Je construis un RAG benchmarké sur FinanceBench, voici ma baseline et mes 3 premiers enseignements") — commence à exister publiquement

**Validation** : un inconnu peut cloner ton repo, lancer `docker-compose up` + 2 commandes, et reproduire ta baseline. La CI est verte.

**Ressources :**
- 📚 FastAPI : https://fastapi.tiangolo.com/tutorial/ (sections 1-10 suffisent)
- 📝 "How to write a great README" : https://github.com/matiassingers/awesome-readme (exemples)
- 🎥 CampusX ou freeCodeCamp "FastAPI tutorial" si besoin d'un guide visuel

---

# 📅 MOIS 2 — RAG avancé + benchmark multi-LLM + observabilité

> Mantra du mois : **"Chaque amélioration est justifiée par un chiffre."** Ton README raconte une progression : baseline → +X → +Y.

## Semaine 5 — Retrieval avancé : hybrid search + reranking + chunking

**Checklist :**
- [ ] Implémenter **BM25** + fusion avec dense (RRF — Reciprocal Rank Fusion) → config "hybrid"
- [ ] Implémenter le **reranking** cross-encoder (bge-reranker-v2-m3) sur le top-20 → top-5
- [ ] Implémenter 2 stratégies de chunking alternatives : **par structure de document** (sections/tableaux via Docling) et **parent-child** (petits chunks pour la recherche, gros chunks pour le contexte)
- [ ] Lancer la matrice d'éval retrieval : {dense, BM25, hybrid} × {avec/sans reranker} × {3 chunkings} → tableau comparatif
- [ ] Mesurer aussi la **latence** ajoutée par chaque étape (le reranker n'est pas gratuit !)
- [ ] ADR : quelle config retrieval gagne et pourquoi

**Validation** : un tableau `docs/benchmark_report.md` montrant la progression du recall@5 baseline → meilleure config, avec les latences. Tu sais expliquer POURQUOI l'hybride aide sur les questions financières (chiffres exacts, noms d'entreprises = lexical ; paraphrases = dense).

**Ressources :**
- 📄 Paper "Searching for Best Practices in RAG" : https://arxiv.org/abs/2407.01219 — LE paper qui benchmarke toutes les composantes, ta lecture la plus rentable du mois
- 📝 Pinecone Learn (série de référence) : https://www.pinecone.io/learn/ (hybrid search, rerankers)
- 📝 Jason Liu, "RAG is more than just embedding search" : https://jxnl.co/writing/
- 🎥 "Advanced RAG techniques" — talks LangChain/LlamaIndex sur YouTube (Lance Martin "Advanced RAG" est excellent)

## Semaine 6 — Benchmark multi-LLM open source

**Objectif** : la pièce maîtresse de ton repo — comparer les LLMs open source à retrieval constant.

**Checklist :**
- [ ] Choisir 4-5 modèles open weights variés en taille : ex. **Qwen 2.5/3 7-14B** (Ollama local), **Llama 3.1 8B** (Ollama/Groq), **Llama 3.3 70B** (Groq), **Mistral Small** (API Mistral free tier), bonus un mini-modèle (Gemma ou Phi) pour le contraste
- [ ] Grâce à LiteLLM, le swap = un changement de config YAML. Vérifie que tes prompts marchent sur tous (les petits modèles sont sensibles au format)
- [ ] Lancer le benchmark complet : meilleure config retrieval × 5 LLMs × 150 questions → accuracy (juge), faithfulness (Ragas sur sous-ensemble), **latence p50/p95**, **coût estimé par question** (tokens × prix, ou 0 pour le local)
- [ ] Construire LE tableau qualité/latence/coût + un graphique scatter (qualité vs coût) — c'est l'argument massue en entretien
- [ ] Analyser : où les petits modèles échouent-ils ? (raisonnement multi-étapes ? extraction de chiffres ?) → 1 page d'analyse

**Validation** : `docs/llm_benchmark.md` avec tableau + graphique + analyse. Tu peux répondre à "quel modèle choisirais-tu pour un client et pourquoi ?" avec des chiffres.

**Ressources :**
- 📚 Ollama : https://ollama.com | Groq : https://console.groq.com/docs
- 📝 Artificial Analysis (comparateur indépendant qualité/prix/vitesse) : https://artificialanalysis.ai
- 📝 LMSYS Chatbot Arena : https://lmarena.ai (pour situer les modèles)
- 🎥 "Choosing the right LLM for production" — talks de conf (AI Engineer Summit sur YouTube : https://www.youtube.com/@aiDotEngineer — chaîne à suivre absolument)

## Semaine 7 — Observabilité (Langfuse) + tests d'évaluation en CI

**Objectif** : le passage "démo → système". LLMOps concret.

**Checklist :**
- [ ] Intégrer **Langfuse** : chaque requête tracée (étapes retrieval, prompts, tokens, latence, coût) — self-host via docker-compose ou cloud free tier
- [ ] Brancher tes scores d'éval dans Langfuse (scores attachés aux traces)
- [ ] Créer un **smoke eval en CI** : 15 questions représentatives qui tournent sur chaque PR (modèle rapide/gratuit) — si l'accuracy chute sous un seuil, la CI échoue. C'est le concept de **régression d'éval**, très rare chez les candidats
- [ ] Ajouter la gestion d'erreurs production : retries (tenacity), timeouts, fallback de modèle (si Groq down → Mistral), rate limiting
- [ ] Caching des réponses LLM pour l'éval (économise tes quotas — un simple cache disque keyed sur hash(prompt+modèle))

**Validation** : tu ouvres Langfuse et tu vois chaque requête décomposée avec coûts. Une PR qui dégrade la qualité est bloquée par la CI. 

**Ressources :**
- 📚 Langfuse : https://langfuse.com/docs (quickstart + scores + datasets)
- 📝 Blog Langfuse "LLM Observability" + leurs exemples RAG
- 📝 Eugene Yan "Evaluating LLM systems in production" : https://eugeneyan.com
- 🎥 Langfuse YouTube channel (démos courtes)

## Semaine 8 — Dashboard de reporting + consolidation

**Checklist :**
- [ ] **Streamlit "Benchmark Explorer"** : charge les JSON de `benchmarks/`, permet de filtrer par config/modèle, affiche tableaux + graphiques (qualité/latence/coût), et un onglet "explorateur d'échecs" (question, réponse gold, réponse modèle, chunks récupérés)
- [ ] Mettre à jour le README : section résultats avec la progression complète mois 1→2
- [ ] Écrire ton **premier article de blog** (Medium/dev.to/LinkedIn) : "Benchmarking open-source LLMs on financial RAG: what 150 questions taught me" — recycle ton `docs/llm_benchmark.md`
- [ ] Rétro mois 2 + préparer le terrain agentic : identifier dans tes échecs les **questions multi-étapes / multi-documents** où le RAG simple plafonne → c'est ta justification mesurée du mois 3
- [ ] 🎯 Rappel certif cloud : si tu suis le plan global, ta prépa AWS SAA tourne en parallèle (~4h/sem prises sur d'autres créneaux) — passage prévu fin mois 3

**Validation** : démo Streamlit fonctionnelle (déployable sur HF Spaces ou Streamlit Cloud gratuitement). Article publié. Liste chiffrée des cas d'échec qui motivent l'agentic.

---

# 📅 MOIS 3 — Agentic RAG, multi-agent, déploiement final

> Mantra du mois : **"L'agentic doit prouver sa valeur."** Chaque pattern ajouté est comparé au RAG simple sur les mêmes métriques (+ coût en tokens et latence, sois honnête sur le trade-off).

## Semaine 9 — Fondamentaux LangGraph + premier pattern : Router + Grader

**Checklist :**
- [ ] Apprendre les primitives LangGraph : StateGraph, nodes, edges conditionnels, checkpointing (1 journée max de tutos)
- [ ] Pattern 1 — **Routing** : un node qui classifie la question (simple lookup / calcul / multi-document) et route vers le bon pipeline
- [ ] Pattern 2 — **Self-correction (style CRAG/Self-RAG)** : node "grader" qui évalue la pertinence des chunks récupérés → si insuffisant, **query rewriting** et nouvelle tentative (boucle, max 2 itérations)
- [ ] Tracer tout dans Langfuse (les graphes agentiques sont durs à débugger sans observabilité — tu vas le constater)
- [ ] Évaluer : agentic v1 vs meilleur RAG simple, sur les 150 questions + spécifiquement sur tes cas d'échec identifiés. Mesure aussi tokens consommés et latence (l'agentic coûte plus cher — chiffre-le !)

**Validation** : tableau comparatif RAG simple vs agentic v1 : qualité, latence, coût. Tu sais dire sur QUELS types de questions l'agentic gagne (et où il ne sert à rien).

**Ressources :**
- 📚 Tutoriel officiel Agentic RAG LangGraph : https://docs.langchain.com/oss/python/langgraph/agentic-rag
- 📚 LangChain Academy (gratuit) : "Introduction to LangGraph" : https://academy.langchain.com
- 📄 Papers : Self-RAG https://arxiv.org/abs/2310.11511 | CRAG https://arxiv.org/abs/2401.15884 (lecture en diagonale : retiens les idées, pas les détails)
- 📝 **LE blog à lire absolument** : Anthropic, "Building Effective Agents" : https://www.anthropic.com/research/building-effective-agents — la référence sur "workflow vs agent", cité partout en entretien
- 🎥 Playlist CampusX "Agentic AI using LangGraph" : https://www.youtube.com/playlist?list=PLKnIA16_RmvYsvB8qkUQuJmJNuiCUJFPL
- 🎥 Série live-coding "Building Agentic RAG with LangGraph" (BigData Boutique, YouTube) — du vrai build avec les vrais problèmes

## Semaine 10 — Query decomposition + extension du golden set multi-hop

**Checklist :**
- [ ] Créer **30 questions multi-documents/multi-étapes** à la main (ex : "Compare la marge brute de 3M et Pepsi en 2022", "L'EBITDA de X a-t-il progressé plus vite que son CA entre 2021 et 2022 ?") avec réponses gold calculées par toi → `data/golden_set/multihop_extension.jsonl`
- [ ] Pattern 3 — **Query decomposition** : node qui décompose la question en sous-questions → retrieval par sous-question → synthèse finale
- [ ] Pattern 4 — **Tool use** : donner à l'agent un outil **calculatrice** (les LLMs sont mauvais en arithmétique — montre que tu le sais) et un outil "lookup métrique dans tableau"
- [ ] Benchmark sur l'extension multi-hop : RAG simple (qui doit échouer) vs agentic avec décomposition (qui doit gagner) — c'est TON narratif chiffré de justification de l'agentic

**Validation** : sur les 30 questions multi-hop : RAG simple X% vs agentic Y% (avec Y >> X espéré). Honnêteté : note aussi les cas où l'agent part en vrille (boucles, sur-décomposition).

**Ressources :**
- 📄 MultiHop-RAG (pour t'inspirer du format de questions) : https://arxiv.org/abs/2401.15391
- 📝 LangChain blog, posts "query decomposition" / "RAG from scratch part 10+" (Lance Martin)
- 🎥 Krish Naik "Agentic RAG with LangGraph" (implémentation détaillée pas à pas)

## Semaine 11 — Workflow multi-agent + serving vLLM (ton atout)

**Checklist :**
- [ ] **Multi-agent supervisé** (pattern supervisor de LangGraph) : un agent "analyste financier" (retrieval + extraction), un agent "vérificateur" (recoupe les chiffres avec les sources, flag les incohérences), un superviseur qui orchestre. Garde-le SIMPLE — 3 agents max, le but est de démontrer le pattern proprement, pas de faire de la science-fiction
- [ ] Évaluer le multi-agent vs agentic simple : la vérification réduit-elle les hallucinations de chiffres (faithfulness) ? À quel coût (tokens ×2-3) ?
- [ ] **Session vLLM** (loue un GPU quelques heures : RunPod/Lightning, ~10-15€) : servir Qwen 7-14B avec vLLM, mesurer throughput/latence vs Ollama, tester prefix caching — écris `docs/serving_benchmark.md`. **C'est la section où ton background optimisation d'inférence brille** : parle de KV cache, batching continu, quantization en connaisseur
- [ ] ADR final : architecture retenue pour le déploiement (quel pipeline par défaut, quels fallbacks)

**Validation** : tableau multi-agent vs single-agent vs RAG simple (qualité/coût/latence). Rapport serving vLLM avec chiffres throughput. 

**Ressources :**
- 📚 LangGraph multi-agent : https://docs.langchain.com/oss/python/langgraph (section multi-agent / supervisor)
- 📚 vLLM docs : https://docs.vllm.ai (quickstart + benchmarking)
- 📝 Anthropic "Building Effective Agents" (relis la partie "quand NE PAS faire d'agents" — cite-la en entretien, ça montre ta maturité)
- 🎥 AI Engineer Summit talks sur le multi-agent en production (chaîne @aiDotEngineer)

## Semaine 12 — Déploiement final, documentation, packaging carrière

**Checklist :**
- [ ] **Déploiement** : docker-compose complet (API FastAPI + Qdrant + Langfuse) déployable en une commande + démo publique sur **HF Spaces** (UI Gradio, pipeline économe : Groq/Mistral free tier) — éventuellement un petit VPS (Scaleway/OVH, signal "cloud français" sympa) si tu veux une URL custom
- [ ] README final niveau vitrine : schéma d'archi complet, GIF de démo, tableau de progression baseline→final, liens vers les rapports de benchmark, badges CI
- [ ] `docs/` complet : tous les ADR, tous les rapports, un "lessons learned"
- [ ] **Article de blog n°2** : "From naive RAG to multi-agent: a measured journey on FinanceBench" — ton storytelling d'entretien par écrit
- [ ] Préparer le **pitch oral de 5 min** du projet (architecture, 3 décisions clés, 3 chiffres, 1 échec instructif) — enregistre-toi
- [ ] Mettre à jour CV + LinkedIn avec le projet (lien démo + repo en haut)
- [ ] Rétro finale : qu'est-ce qui manque ? (idées d'extensions : MCP server, feedback utilisateur dans Langfuse, A/B de prompts...) → section "Roadmap" du README (montre que tu penses produit)

**Validation finale du projet (la checklist du recruteur) :**
- [ ] Démo live accessible par URL ✅
- [ ] Repro complète en 3 commandes ✅
- [ ] CI verte avec tests + smoke eval ✅
- [ ] Benchmark multi-LLM open source avec qualité/latence/coût ✅
- [ ] Progression chiffrée naive → advanced → agentic → multi-agent ✅
- [ ] Observabilité Langfuse démontrable ✅
- [ ] 2 articles publiés + posts LinkedIn ✅

---

## 📌 Récap des 10 ressources à mettre en favoris dès aujourd'hui

1. FinanceBench (repo + data) : https://github.com/patronus-ai/financebench
2. Hamel Husain — Evals : https://hamel.dev/blog/posts/evals/
3. Paper "Searching for Best Practices in RAG" : https://arxiv.org/abs/2407.01219
4. Anthropic — "Building Effective Agents" : https://www.anthropic.com/research/building-effective-agents
5. Tutoriel Agentic RAG LangGraph officiel : https://docs.langchain.com/oss/python/langgraph/agentic-rag
6. Ragas : https://docs.ragas.io
7. Langfuse : https://langfuse.com/docs
8. LiteLLM : https://docs.litellm.ai
9. Playlist "RAG from Scratch" (LangChain/freeCodeCamp) : https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x
10. Chaîne AI Engineer Summit : https://www.youtube.com/@aiDotEngineer

## ⚠️ Les 5 pièges qui te guettent (toi spécifiquement, ex-chercheur)

1. **Perfectionnisme de baseline** : ne passe pas 3 semaines sur le parsing PDF. "Assez bon + mesuré" > "parfait + en retard". Tu itéreras.
2. **Tutorial hell** : max 3h de contenu/semaine. Le reste = code.
3. **Sur-ingénierie agentique** : un agent qui boucle 5 fois pour gagner 1 point de recall n'impressionne personne. Le trade-off honnête, si.
4. **Repo silencieux** : commit + push chaque session. Un historique de commits régulier sur 3 mois EST un signal pour les recruteurs.
5. **Reporter la visibilité** : les posts LinkedIn dès le mois 1, pas "quand ce sera fini". Le réseau se construit pendant, pas après.

Bon courage Ahmed — dans 12 semaines, tu auras un repo que 99% des candidats AI Engineer n'ont pas. 🚀