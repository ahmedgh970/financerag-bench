# ADR 0004 — Workflow CRAG : grille ancrée dans l'evidence, advanced 12K / 24K vs grading

## Statut

Accepté.

## Contexte

`src/workflow/` est un graphe LangGraph **déterministe** : un seul code path dont
les nœuds s'activent par YAML (`configs/workflow/`). Deux lignes de l'ablation
sont mesurées ici :

- **advanced** — `retrieve → generate`, aucun nœud optionnel. Son prompt est
  identique à celui du pipeline naïf.
- **grading** — `retrieve → grade → generate`. Le grader note **chaque passage**
  de 0 à 3 (échelle UMBRELA, un appel LLM par passage), garde ceux notés ≥ 2
  (`keep_threshold: 2`) et complète toujours jusqu'à 3 passages par le rang
  (`min_chunks: 3`).

Tout le reste est fixé pour que seule la ligne varie :

- **Retrieval rejoué** : `reranked(dense)`, prefetch 50, k = 20, doc-scoped
  (ADR 0001), matérialisé une fois puis rejoué (`retriever: replay`). Toutes les
  lignes voient **exactement les mêmes 20 passages**, dans le même ordre.
- **Générateur** : `granite4.1:8b` (ADR 0002), `temperature: 0`,
  `max_tokens: 1024`.
- **Fenêtre de contexte épinglée** (`num_ctx`) : le contexte est tronqué **par la
  fin** pour tenir dans la fenêtre. À 12 288 tokens, advanced garde ~10 passages
  sur 20 ; à 24 576, 19,7 en moyenne (135 questions sur 150 gardent les 20).

Les trois runs comparés : advanced à `num_ctx` 12 288 et 24 576, grading à
12 288.

### Pourquoi une nouvelle grille

La grille `correct` / `grounded` d'ADR 0002 ne dit pas **si le modèle avait
l'evidence sous les yeux**. On ne peut donc pas séparer un échec de retrieval
d'un échec de génération, ni une hallucination d'un refus honnête. La nouvelle
grille classe chaque réponse dans **une** des quatre catégories :

| Catégorie | Définition |
|---|---|
| **Good job** | Réponse correcte **et** evidence gold dans le prompt — ou chiffres vérifiés dans un autre passage récupéré (*alt evidence*, compté à part) |
| **Hallucinating** | Répond (juste ou faux) **sans** l'evidence dans son contexte |
| **Need help** | Répond faux **alors que** l'evidence était dans le prompt (échec de génération) |
| **Dont know** | Refus / « le contexte ne contient pas… » (prioritaire ; on note à côté si l'evidence était présente) |

**Evidence présente (critère A)** : chaque evidence gold est résolue sur sa page
physique par `src/evaluation/common/matching.py`, la **même** définition que le
recall@k du retrieval (ADR 0001). Un passage de **chaque** page gold doit figurer
dans les `sources` réellement envoyées au générateur. C'est calculé par code, sans
seuil. Un diagnostic de recouvrement mots / chiffres avec la page gold a été
essayé puis retiré. Il ne repérait que 2 des 3 faux positifs connus (Q17, Q148 ;
Q10 manqué) et levait des alertes sur des Good job correctement fondés (Q16, Q43,
Q54, Q149), parce que l'evidence gold couvre une page entière alors qu'une ligne
suffit.

**Juge** : Claude (frontier, comme ADR 0002). Chaque question est jugée une par
une : réponse **entière**, vérifiée contre les **passages du prompt**, avec le
verdict des autres runs sous les yeux pour rester cohérent. Le juge ne fournit
que `correct` / `refused` / `alt_supported` + une justification spécifique ;
`src/evaluation/judge/grid.py` calcule l'evidence et la catégorie.

**Règles de cohérence appliquées aux trois runs** :

- Numérique strict (7.8 vs 7.9 = faux ; erreur de magnitude = faux). Une
  question oui / non tolère un arrondi si la direction est juste.
- Bon label obtenu par un raisonnement faux ou contradictoire alors que
  l'evidence était là = **faux** (ex. Q77, Q94, Q109 à 24K).
- Bon chiffre via une métrique adjacente sans evidence = Hallucinating, pas
  d'*alt evidence* (ex. Q98 : bénéfice avant impôt pris pour le bénéfice net).
- Argument du silence (« aucune source n'en parle, donc non ») = **refus**.
- Liste : un item faux = incorrect (Q133 à 24K : 2/3).

## Résultats

150 QA, `granite4.1:8b`, retrieval rejoué k = 20, un run par configuration :

| Ligne | Good job | Hallucinating | Need help | Dont know | Correct | Evidence dans le prompt | Passages envoyés | Latence / Q (moy · p90) | Appels LLM / Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| advanced, `num_ctx` 12 288 | 79 (52,7 %) | 32 | 16 | 23 | 86 | 93 | 10,2 | 107 s · 210 s | 1 |
| grading, `num_ctx` 12 288 | 83 (55,3 %) | 31 | 14 | 22 | 88 | 92 | 5,6 | 161 s · 250 s | 20,9 |
| **advanced, `num_ctx` 24 576** | **88 (58,7 %)** | **21** | 22 | **19** | **92** | **105** | 19,7 | 187 s · 334 s | 1 |

- **Alt evidence** parmi les Good job : 6 (adv 12K), 9 (grading), 9 (adv 24K).
- **Refus avec l'evidence dans le prompt** : 4 par run.
  - adv 12K : Q56, Q74, Q110, Q148.
  - grading : Q24, Q76, Q92, Q134.
  - adv 24K : Q92, Q138, Q141, Q148.
- **Latence du grading décomposée** : le grader prend 79 s par question en
  moyenne, la génération 82 s (contre 107 s pour advanced 12K, qui génère sur
  deux fois plus de passages).

Transitions advanced 12K → 24K (même retrieval, seule la fenêtre change) :

| | Nombre | Questions |
|---|---:|---|
| Gains (→ Good job) | 16 | Q21, 30, 31, 40, 49, 58, 70, 74, 86, 88, 110, 116, 117, 127, 144, 149 |
| — dont evidence nouvellement présente | 4 | Q21, 40, 116, 149 |
| Pertes (Good job →) | 7 | Q72, 81, 94, 112 (Need help) · Q92, 138, 141 (Dont know) |
| Evidence gagnée / perdue | +12 / 0 | la fenêtre 24K contient toujours celle de 12K |

Grading vs advanced 12K : evidence gagnée sur 6 questions (Q14, 21, 43, 113,
116, 135) et **perdue sur 7** (Q18, 68, 85, 105, 138, 141, 148). Advanced 24K bat
grading sur 10 questions et perd sur 5.

## Analyse

**1. Advanced 24K est la meilleure ligne, mais pas surtout grâce à l'evidence
ajoutée.** La fenêtre doublée fait passer l'evidence présente de 93 à 105
questions, sans en perdre aucune. Pourtant, seuls **4 des 16 gains** viennent
d'une evidence nouvellement présente. Les 12 autres questions avaient déjà
l'evidence à 12K : le modèle a simplement répondu autrement avec plus de
contexte autour. La génération est **sensible à des changements qui ne
touchent pas l'information utile**. Q81 le montre à l'état pur : un 8-K court,
le même prompt de 9 passages dans les deux runs, et la réponse passe de juste à
fausse. On ne peut pas mesurer cette sensibilité avec les données actuelles
(seules 3 questions ont un prompt identique entre 12K et 24K). **Un écart de
±5 questions entre deux lignes n'est donc pas interprétable** comme un gain
structurel.

**2. Plus de contexte déplace les échecs de Hallucinating vers Need help.** À
24K, Hallucinating tombe de 32 à 21 parce que l'evidence est plus souvent là, mais
Need help monte de 16 à 22 : **26 questions ont l'evidence dans le prompt sans
être converties** (22 Need help + 4 refus). Le goulot est maintenant **le
générateur**. Typologie des échecs avec evidence présente :

- **Arithmétique / arrondi** : moyenne recalculée faux (Q72), arrondi
  intermédiaire (Q76, Q105).
- **Mauvaise définition ou mauvais poste** :
  - amortissement du contenu ajouté au D&A (Q113) ;
  - passif total pris pour la dette (Q112) ;
  - SG&A total pris pour les salaires (Q142) ;
  - bénéfice d'exploitation GAAP + impôt comme « EBIT ajusté » (Q108).
- **Label juste, raisonnement invalide** (non crédité) : Q77, Q94, Q109.
- **Conclusion contraire à ses propres chiffres** : Q14 (« en amélioration »
  alors qu'il calcule 36,8 % → 34,6 %), Q81, Q89.
- **Tableau le plus fin ignoré** : Q135 (s'arrête à US / International alors
  que le tableau par région est là).
- **Refus avec l'evidence sous les yeux** : Q92, Q138 (argument du silence),
  Q141.

**3. Le grading améliore la génération mais coûte cher et perd de
l'evidence.** Réduire le contexte à 5,6 passages retire du bruit : la plupart de
ses gains sur advanced 12K se font à evidence égale. Mais le grader **écarte
l'evidence sur 7 questions** (Q138 et Q141 : la page gold est filtrée et le
modèle refuse honnêtement). Surtout, il coûte 20,9 appels LLM et **79 s par
question**, soit la moitié de la latence de la ligne. Net : +4 Good job sur
advanced 12K, −5 sur advanced 24K.

**4. Le plafond du retrieval reste le premier facteur.** Même à 24K, **45
questions n'ont pas l'evidence dans le prompt**. Sur ces 45, 9 réussissent
quand même par *alt evidence* (le MD&A répète souvent les états financiers),
15 refusent (dont la plupart honnêtement) et 17 répondent faux. Sur les trois
runs, **97 questions** sont réussies au moins une fois et **67** par les trois :
l'écart entre 88 et 97 est ce qu'une meilleure génération peut encore gagner à
retrieval constant.

## Décision

1. **La grille à 4 catégories devient la métrique de référence** des lignes du
   workflow : evidence calculée par code (critère A, même définition que le
   recall@k), verdicts `correct` / `refused` / `alt_supported` par le juge. La
   grille `correct` / `grounded` d'ADR 0002 reste la référence du pipeline naïf ;
   les deux ne sont **pas directement comparables** (le 65,3 % `equivalent`
   d'ADR 0002 est une autre métrique).
2. **Advanced `num_ctx` 24 576 est la ligne de référence qualité** (88 / 150,
   1 appel LLM), au prix de 187 s par question et de plus de Need help.
3. **Le grading n'est pas adopté en l'état** : 20,9 appels et 161 s par
   question pour un résultat inférieur à advanced 24K.
4. **Hypothèse à tester ensuite** : un grader optimisé (moins cher, sans perte
   d'evidence) couplé à un prompt de génération qui **exploite mieux l'evidence**
   (calcul explicite, définitions des postes, pas de refus par argument du
   silence) peut atteindre au moins advanced 24K avec un contexte court, donc à
   latence moindre. C'est une hypothèse, pas un résultat ; elle sera mesurée
   avec cette même grille.
5. **Un écart de moins de ~5 questions entre deux lignes est lu comme non
   significatif** tant que la sensibilité de la génération n'est pas mesurée.

## Conséquences

- **Code** :
  - `src/evaluation/judge/grid.py` : evidence présente (critère A), règle de
    catégorie, et jointure réponses / verdicts / golden set, en échec si un
    verdict manque ;
  - `src/evaluation/run_judge.py grid` : écrit la grille et affiche ses
    comptes ;
  - `src/evaluation/common/matching.py` : résolution des pages gold, partagée
    avec l'évaluation du retrieval ;
  - `scripts/judge_view.py` : vue de jugement par question (réponses entières,
    sources, passages) ;
  - `configs/evaluation/judge/evidence_grid.yaml` : une seule config pour tous
    les runs, le run étant choisi par `make grid ANSWERS=<answers.jsonl>` ; les
    verdicts sont retrouvés d'après le nom du fichier de réponses.
- **Données** (locales, non versionnées) :
  - verdicts dans
    `data/processed/judged/verdicts/{answers_stem}.claude.jsonl` ;
  - grille dans `data/processed/judged/{answers_stem}_grid_by_claude.jsonl`.
- **Limites assumées** :
  - **Juge frontier non re-runnable** (cf. ADR 0003). Les verdicts sont
    archivés question par question avec leur justification.
  - **Critère A imparfait dans les deux sens.**
    - *Page gold présente mais passage utile absent* : Q10, Q17, Q148 à 24K,
      comptés « evidence présente ».
    - *Page gold absente mais evidence ailleurs* : couvert par l'*alt
      evidence*.
    - *Gold multi-pages dont une seule suffit* : Q144.
  - **Passes indulgentes** documentées dans les justifications : Q41, Q42,
    Q132 (les trois runs), Q112 (runs 12K), Q86 (advanced 12K), Q20 et Q35
    (advanced 24K).
  - **Gold discutables** : Q119 (définition non standard du working capital),
    Q134 (77,78 contre ~70 correct).
  - **Troncature à `max_tokens` 1024** : Q82 et Q131 à 24K. Q131 finit sans
    conclusion et compte comme Dont know.
  - **Un seul run par configuration**, variance de génération non mesurée.
    Latences mesurées sur le GPU portable 8 Go (valables en relatif).
- **Suite** : optimiser le grader (coût et rappel de l'evidence) et le prompt de
  génération, puis re-mesurer contre advanced 24K avec cette grille.
