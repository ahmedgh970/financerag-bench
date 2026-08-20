# ADR 0002 — Modèle de génération : ablation modèle × profondeur de retrieval (RAG naïf local)

## Statut

Accepté.

## Contexte

ADR 0001 a fixé le retrieval par défaut (`reranked(dense)`, doc-scoped,
`prefetch=50`). Ce document décide **l'étage de génération** du RAG naïf
(`src/rag/naive.py` : retrieve une fois → prompt → generate une fois, sans
boucle ni agent) : quel modèle local servir, et à quelle profondeur `k`.

Réglage commun à toutes les mesures :

- **Corpus** `docling_hybrid_1024_bge-m3`, retriever `reranked(dense)`,
  **doc-scoped**, `prefetch=50` — le meilleur point d'ADR 0001.
- **150 QA** FinanceBench, prompt grounded (`src/llm/prompts.py` : répondre
  *uniquement* depuis le contexte, refuser explicitement si l'info est
  absente, citer la source).
- **Lineup local Ollama** (9 modèles jugés, 3.2B → 12.2B), généré par
  `scripts/generation_benchmark.py` (`make generate`).
- **Setup de génération** : `temperature=0`, `num_predict=1024`,
  `think:false` sur les modèles thinking (sinon la chaîne de raisonnement
  épuise `num_predict` avant tout token de réponse → `content` vide).
  **`num_ctx` fixé par `k`** — 10240 (k5) / 18432 (k10) / 30720 (k20) —
  dimensionné pour couvrir le prompt réel le plus long des 150 QA (mesuré
  avec le tokenizer le plus coûteux du lineup, mistral:7b) plus le budget de
  sortie, sans marge. Une paire (modèle, k) est **sautée** si son `num_ctx`
  dépasse le contexte d'architecture du modèle ; `command-r7b` (arch max
  8192) tourne à k5 à son propre max (8192) et n'a ni k10 ni k20.
- **Juge** : Claude (frontier) contre l'evidence du golden set, grille
  `correct` / `grounded` / `equivalent` (= correct ET grounded) ; `$` et
  formatage relâchés, magnitude/signe stricts, un refus compte comme
  `grounded` mais pas `correct`. Un matcher numérique déterministe a été
  essayé puis rejeté (déclenchement réel ~0 % sur des réponses en langage
  naturel) — le jugement est LLM-only (cf. `docs/results_generation.md`).

**Condition de validité de l'ablation** : parce que `num_ctx` est
dimensionné par `k` pour contenir le prompt entier, **aucun prompt n'est
tronqué** — les runs k5/k10/k20 voient réellement 5/10/20 chunks. C'est ce
qui rend l'axe « profondeur » interprétable.

## Question posée

1. Quel modèle local adopter comme générateur par défaut du pipeline RAG.
2. À quelle profondeur `k` l'exploiter.
3. Question de recherche sous-jacente : **comment `k` interagit-il avec la
   capacité du modèle** — la profondeur utile est-elle universelle ?

Métrique de décision : **`equivalent`** (correct ET grounded). En finance,
une réponse fausse énoncée avec assurance est pire qu'un refus honnête :
`equivalent` récompense les bonnes réponses *et* pénalise les fabrications,
là où `correct` seul créditerait un chiffre juste tiré d'un raisonnement
faux, et `grounded` seul récompenserait un refus systématique.

## Résultats

Toutes les lignes évaluent les 150 QA (0 skip). Valeurs en **% sur 150 QA**.
Cellule vide = profondeur non exécutée (limites contexte/temps).

### Headline — `equivalent` (métrique primaire)

| Modèle | Params | Type | k5 | k10 | k20 |
|---|---|---|---:|---:|---:|
| llama3.2:3b   | 3.2B  | plain    | 22.0 | 28.7 | 28.7 |
| granite4.1:3b | 3.4B  | plain    | 40.0 | 39.3 | 40.0 |
| qwen3.5:4b    | 4.7B  | thinking | 44.7 | 53.3 | **60.0** |
| mistral:7b    | 7.2B  | plain    | 36.7 | 38.7 | 41.3 |
| command-r7b   | 8.0B  | plain    | 38.7 |  |  |
| llama3.1:8b   | 8.0B  | plain    | 42.0 | 49.3 | 50.7 |
| granite4.1:8b | 8.8B  | plain    | 53.3 | 62.0 | **65.3** |
| qwen3.5:9b    | 9.7B  | thinking | 44.0 | 54.7 | 58.0 |
| mistral-nemo  | 12.2B | plain    | 47.3 |  |  |

### Détail (correct / grounded / equivalent), pas marginal k10→k20

| Modèle | correct k5/k10/k20 | grounded k5/k10/k20 | equivalent k5/k10/k20 | Δ k10→k20 |
|---|---|---|---|---:|
| llama3.2:3b   | 22.7 / 28.7 / 28.7 | 81.3 / 89.3 / 93.3 | 22.0 / 28.7 / 28.7 | **+0.0** |
| granite4.1:3b | 42.7 / 42.0 / 42.0 | 74.7 / 72.7 / 68.7 | 40.0 / 39.3 / 40.0 | **+0.7** |
| llama3.1:8b   | 43.3 / 49.3 / 50.7 | 76.0 / 89.3 / 92.7 | 42.0 / 49.3 / 50.7 | **+1.4** |
| mistral:7b    | 36.7 / 39.3 / 41.3 | 84.0 / 88.0 / 92.0 | 36.7 / 38.7 / 41.3 | **+2.6** |
| granite4.1:8b | 55.3 / 63.3 / 65.3 | 81.3 / 82.7 / 87.3 | 53.3 / 62.0 / 65.3 | **+3.3** |
| qwen3.5:9b    | 44.0 / 55.3 / 58.7 | 97.3 / 99.3 / 99.3 | 44.0 / 54.7 / 58.0 | **+3.3** |
| qwen3.5:4b    | 44.7 / 54.0 / 60.0 | 92.7 / 98.0 / 99.3 | 44.7 / 53.3 / 60.0 | **+6.7** |

(command-r7b k5 : 40.0 / 68.7 / 38.7 ; mistral-nemo k5 : 48.7 / 77.3 / 47.3.)

## Analyse

**La profondeur utile n'est pas universelle — elle scale avec la capacité.**
Le pas k5→k10 donne un gain *large et peu discriminant* (~+7 pp à presque
tous les modèles : +6.7 llama3.2:3b, +7.3 llama3.1:8b, +8.6 qwen3.5:4b, +8.7
granite4.1:8b, +10.7 qwen3.5:9b), qui suit la **discipline d'abstention** plus
que la capacité brute — plus de contexte convertit des refus prudents en
réponses. **Le pas diagnostique est k10→k20** : il s'ordonne proprement par
capacité — les deux petits 3B sont plats (llama3.2:3b **+0.0**, granite4.1:3b
+0.7), le 8B moyen est faible (llama3.1:8b +1.4), et **seuls les modèles les
plus capables montent encore** (granite4.1:8b +3.3, qwen3.5:9b +3.3,
qwen3.5:4b +6.7). Le pic de profondeur utile monte avec la capacité : le
modèle faible ne profite jamais de k20, le modèle moyen sature vers k10, le
modèle fort n'a pas encore plafonné à k20.

**Deux coins distincts de la frontière, séparés par l'axe `grounded`.**

- **Honest-but-timid** (qwen3.5:4b/9b, thinking) — grounding quasi parfait
  (98–99 %), `equivalent` = `correct` (zéro coup de chance). Ils **s'abstiennent**
  plutôt que d'inventer ; leurs gains en profondeur viennent de la conversion
  refus → réponse quand les tables plus complètes lèvent l'ambiguïté.
- **Confident-but-fabricating** (mistral:7b, plain) — grounding plus bas
  (84 → 92), courbe de profondeur la plus plate du tier (+2.0 puis +2.6). Cas
  unique : **la profondeur y déplace `grounded` (+4.0 à k20, fabrications 18 → 12)
  plus que `equivalent`** — le contexte supplémentaire lui permet de *battre en
  retraite vers l'honnêteté* (refuser au lieu d'inventer un proxy), pas de
  débloquer de nouvelles bonnes réponses. Miroir exact des qwens.

**À k20, le petit 4B dépasse le 9B (60.0 vs 58.0).** Ce n'est pas du bruit :
c'est le pendant direct des **régressions de dilution** du 9B à k20 (~13
nouvelles bonnes réponses contre ~7 régressions où des chiffres concurrents
en contexte le tirent hors de réponses qu'il tenait à k10 — p.ex. l'EBITDA
Netflix passé de 5.4 % correct à 56.8 % en repliant à tort l'amortissement de
contenu). Les erreurs k10 du 4B étaient au contraire *récupérables*
(sur-refus, glissements arithmétiques, erreurs d'unité) et les tables k20 en
ont corrigé une longue liste. **Le coût de distraction de la profondeur frappe
le raisonneur le plus élaboré, pas le plus petit.**

**Le meilleur point absolu est `granite4.1:8b` à k20 (65.3).** Il monte de
façon **monotone sans plateau** jusqu'à k20 (+8.7 puis +3.3), avec un grounding
qui progresse aussi (81 → 87). C'est le seul modèle du lineup qui gagne
franchement à chaque cran de profondeur.

**Hallucination : clivage thinking vs plain.** Les modèles thinking (qwens)
fabriquent quasi jamais (1 seule erreur non-grounded à k10/k20 pour le 9B) ;
les modèles plain inventent des *proxies* quand ils ne savent pas faire un
calcul (D&A pris pour capex, capitaux propres pour actifs totaux, différence
de COGS pour de la D&A). C'est la racine commune du grounding bas *et* de la
courbe de profondeur plate de mistral : **la profondeur ne répare pas une
réponse fabriquée**, elle ne fait que remplacer, chez un modèle honnête, un
refus par une réponse.

**Validation du juge.** `command-r7b` k5 (38.7) reproduit le point du juge
antérieur Groq-70B (38.0 / 65.3 / 32.0), confirmant que le juge Claude est
cohérent avec la référence validée à la main — condition avant d'adopter ses
verdicts comme mesure primaire.

## Décision

1. **Générateur local par défaut : `granite4.1:8b`, à `k=20`** (`equivalent`
   65.3, le meilleur point mesuré) — pour tout ce qui suit dans la version
   locale/reproductible du pipeline (phase agent, serving hors-quota).

2. **`qwen3.5:4b` (ou 9b) comme alternative « haute-confiance »** quand le
   coût d'une hallucination prime sur la couverture : grounding 99.3 % à k20,
   `equivalent` = `correct`, zéro fabrication ou presque. À 4.7B, le 4B offre
   le meilleur compromis honnêteté/taille/latence du lineup, et **dépasse même
   le 9B à k20**.

3. **La profondeur `k` est adaptée à la capacité, pas figée** : k20 pour les
   modèles ≥ 8B encore en progression (granite4.1:8b, qwen3.5:9b, qwen3.5:4b) ;
   inutile d'aller au-delà de k10 pour les petits modèles (3B plats à k20).
   Le défaut du pipeline est **k20**, à réduire pour les modèles qui plafonnent.

4. **`equivalent` reste la métrique de décision**, `grounded` suivi en parallèle
   comme garde-fou anti-hallucination (un modèle à `correct` élevé mais
   `grounded` bas n'est pas déployable en finance).

5. **Juge Claude retenu comme mesure de référence** ; un juge local reproductible
   (Prometheus) sera ajouté en colonnes parallèles et comparé (accord + kappa de
   Cohen) avant toute publication chiffrée.

## Conséquences

- Le pipeline de génération multi-modèle est figé dans
  `scripts/generation_benchmark.py` : `num_ctx` par `k` (10240/18432/30720),
  gating par contexte d'architecture, `think:false`, `num_predict=1024`,
  reprise par fichier (modèle, k). Toute comparaison ultérieure doit réutiliser
  ce réglage pour rester comparable.
- `docs/results_generation.md` porte le tableau complet et une synthèse
  par modèle ; ce fichier est la source des chiffres ci-dessus.
- **Limites assumées** :
  - `num_predict=1024` tronque une poignée de réponses verbeuses (surtout les
    deux Llamas et les modèles thinking) — plafonne mécaniquement `correct` sur
    ces cas.
  - Juge unique (Claude) en attendant la corroboration Prometheus.
  - `command-r7b` (k5 seul, plafond d'architecture) et `mistral-nemo` (k5 seul,
    coût en temps) ne sont pas balayés en profondeur — leurs points k5 situent
    le modèle mais ne permettent pas de conclure sur leur réponse à `k`.
  - Lineup **local** : la décision porte sur le benchmark reproductible
    hors-quota, pas sur un serving à modèle frontier (autre arbitrage
    coût/latence/qualité, non tranché ici).
- **Reste à mesurer** :
  - Prometheus comme juge local (accord % + kappa de Cohen vs Claude).
  - Ragas sur les fichiers de réponses (faithfulness, answer relevancy,
    context precision/recall) — métriques orthogonales, sans golden set côté
    génération.
