# ADR 0001 — Stratégie de retrieval : dense vs BM25 vs hybride RRF

## Statut

Accepté.

## Contexte

Corpus : 368 filings SEC parsés avec Docling, chunkés avec le `HybridChunker`
(budget 512 tokens, tokenizer BGE-M3). Index Qdrant `docling_hybrid_bge-m3`
(183 687 points, embeddings BGE-M3 en fp16). Jeu de questions : les 150 QA
FinanceBench (open-source), chacune rattachée à un `doc_name` (le filing dont
elle provient).

Deux réglages de recherche sont mesurés séparément :

- **global** : le retriever cherche dans les 368 documents (aucune information
  sur le bon document).
- **doc-scoped** : le retriever est restreint au `doc_name` de la question
  (champ fourni par le dataset lui-même — un réglage « document connu »
  standard, pas une fuite).

Métriques : recall@k, precision@k, MRR, nDCG@k (k ∈ {1,3,5,10}), avec
pertinence définie au niveau **page** (page gold résolue par recouvrement
lexical entre l'evidence et le corpus, dédupliquée par page).

## Options considérées

1. **Dense** — BGE-M3 + Qdrant (similarité cosinus).
2. **BM25** — lexical pur (`rank-bm25`), IDF calculé sur tout le corpus.
3. **Hybride** — fusion dense + BM25 par Reciprocal Rank Fusion (RRF).

## Résultats

Toutes les lignes ci-dessous évaluent les 150 QA (0 skip), corpus
`docling_hybrid_bge-m3` (chunker hybrid, 512 tokens).

| # | Retriever | Scope | recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| v0 | dense | global | 0.130 | 0.190 | 0.230 | 0.297 | 0.182 | 0.205 |
| P1 | **dense** | **doc-scoped** | **0.227** | **0.332** | **0.402** | **0.552** | **0.338** | **0.377** |
| P2 | bm25 | doc-scoped | 0.113 | 0.176 | 0.222 | 0.261 | 0.160 | 0.183 |
| P2 | hybride RRF (naïf : pf=100, poids égaux) | doc-scoped | 0.166 | 0.229 | 0.252 | 0.340 | 0.223 | 0.247 |
| P2 | hybride RRF (réglé : pf=20, w_bm25=0.3) | doc-scoped | 0.173 | 0.286 | 0.361 | 0.556 | 0.288 | 0.343 |

Balayage complémentaire (rankings dense/BM25 collectés une fois, fusion
recalculée en mémoire, profondeur de pool `pf` × poids BM25 `w_bm25`,
`recall@10` uniquement) :

| pf | w_bm25=0.25 | w_bm25=0.5 | w_bm25=1.0 |
|---|---|---|---|
| 10  | 0.552 | 0.552 | 0.464 |
| 20  | 0.556 | 0.556 | 0.447 |
| 100 | 0.507 | 0.430 | 0.340 |

(repère : dense seul = 0.552)

## Analyse

**Doc-scoped vs global.** Restreindre la recherche au document gold fait
presque doubler le recall@10 (0.297 → 0.552). Le réglage global mélange deux
sous-tâches — router vers le bon document et retrouver le bon passage —
alors que FinanceBench fournit le document cible. Le doc-scoped isole la
qualité de retrieval de passage ; c'est la métrique retenue comme référence
pour la suite des expériences.

**BM25 est nettement plus faible que le dense sur ce benchmark**
(recall@10 = 0.261 vs 0.552). Les questions FinanceBench sont formulées en
langage naturel et paraphrasent le texte du filing (« capital expenditure »,
« capex », des tournures variables) ; l'embedding sémantique de BGE-M3 capture
mieux cette reformulation que le recouvrement lexical exact de BM25.

**La fusion RRF naïve (pool profond, poids égaux) dégrade le résultat**
en dessous du dense seul (0.340 vs 0.552) : à profondeur pf=100, BM25 injecte
dans le classement fusionné des chunks de mauvaise qualité qui délogent les
bons résultats du dense. Un RRF **réglé** (pool peu profond pf=20, poids BM25
réduit à 0.3) permet de retrouver le niveau du dense, mais sans le dépasser
significativement (0.556 vs 0.552, soit +0.4 point — dans le bruit) — et reste
**inférieur** au dense seul sur recall@1, recall@3, recall@5, MRR et nDCG@10.

## Décision

**Le retriever dense (BGE-M3) reste la référence** du pipeline. BM25 et
l'hybride RRF sont implémentés, testés et enregistrés dans le registry
(`bm25`, `hybrid`) — utilisables pour des ablations futures ou sur un corpus
où le signal lexical (tickers, codes exacts) compterait davantage — mais ne
sont pas adoptés par défaut : ils n'apportent aucun gain mesurable ici et
dégradent certaines métriques.

## Conséquences

- L'harnais d'évaluation gagne un flag `doc_scoped` (`EvalConfig`) ; toute
  nouvelle expérience doit le préciser explicitement pour rester comparable
  aux résultats ci-dessus.
- Le retriever hybride expose désormais `prefetch` et des poids par liste
  (`dense_weight`, `sparse_weight`) plutôt qu'un RRF à poids égaux — le
  réglage par défaut (`prefetch=20`, `sparse_weight=0.3`) correspond au
  meilleur point trouvé lors du balayage.
- Le recall@10 absolu du dense (0.552) reste modeste : la piste retenue pour
  la suite est un **reranker cross-encoder** appliqué sur le top-k du dense,
  plutôt qu'un retriever lexical supplémentaire.
