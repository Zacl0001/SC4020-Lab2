# SC4020 Project 2 — Biased Matrix Factorization

> **Code-only GitHub package:** this branch adds the model, runner, tests,
> README, scientific report, protocol and `.gitignore`. Trained checkpoints,
> notebook, CSVs, figures, logs and provenance archives remain local and are
> **not included in this push**. Paths to those artifacts below describe the
> retained local experiment or outputs of a deliberate reproduction; they are
> not download links. The numeric findings are preserved in this README/report.
> Historical comparison CSVs are optional, so a fresh checkout can reproduce
> the campaign using the existing tracked train/validation splits.

This directory contains the **official consolidated Biased MF implementation and completed scientific campaign** for Task 2. The research question is:

> How do latent-factor dimension and L2 regularization influence prediction accuracy, overfitting and generalization on DAVIS and KIBA?

The campaign comprises **73 physical fits**, including capacity and F–lambda studies, learning-rate refinement, three-seed assessment and three justified epoch-extension reruns. It replaces the earlier standalone baseline/grid/sweep workflow. The core mathematical implementation and shared evaluation definitions are unchanged.

**Both datasets selected F=128, lambda=1, learning rate=0.001, batch size=2048.** Configuration selection uses three-seed mean validation MSE; the selected local checkpoint always uses **seed 42**, never the luckiest seed.

| Published model | Training MSE | Validation MSE | CI | Hit@1 | Hit@5 | Best / stop epoch |
|---|---:|---:|---:|---:|---:|---|
| DAVIS | 0.0262192794 | **0.2598417435** | 0.8731753721 | 0.3382352941 | 0.7205882353 | 74 / 84 |
| KIBA | 0.0232474522 | **0.1887607637** | 0.8480527495 | 0.6903914591 | 0.9522114896 | 34 / 44 |

These are **validation results, not final test performance**. Full-precision values remain in `results/biased_mf/final_selection.csv`. Read the [scientific report](results/biased_mf/REPORT.md), [protocol](results/biased_mf/PROTOCOL.md), or `biased_mf.ipynb` for the complete evidence.

Numbers labeled **illustrative** below are teaching examples, not additional experiments. All file links are relative to `task2/`; terminal commands start from the repository root.

## Contents

1. [Objective and scope](#objective-and-scope)
2. [Datasets, preparation and splits](#datasets-preparation-and-splits)
3. [Prediction equation and matrix dimensions](#prediction-equation-and-matrix-dimensions)
4. [Squared error, objective and regularization](#squared-error-objective-and-regularization)
5. [Gradients and Adam](#gradients-and-adam)
6. [Training and early stopping](#training-and-early-stopping)
7. [Scientific experimental campaign](#scientific-experimental-campaign)
8. [Evaluation metrics and candidate counts](#evaluation-metrics-and-candidate-counts)
9. [Final selection and interpretation](#final-selection-and-interpretation)
10. [Code and artifact guide](#code-and-artifact-guide)
11. [Correctness tests and verification](#correctness-tests-and-verification)
12. [WSL commands](#wsl-commands)
13. [Consolidation, provenance and exclusions](#consolidation-provenance-and-exclusions)
14. [Limitations and remaining work](#limitations-and-remaining-work)
15. [Glossary and references](#glossary-and-references)

## Objective and scope

A **ligand**, **drug** or **compound** is a small molecule. A **target** is a protein. Binding affinity measures how strongly the molecule binds to the protein under the measurement conditions. Task 2 hides measured drug–target scores and predicts them, allowing target ranking for a ligand already represented in training.

- **DAVIS** is a kinase-inhibitor binding benchmark. Task 2 uses already-converted pKd scores, where higher means stronger affinity.
- **KIBA** combines bioactivity information from measurements such as Ki, Kd and IC50. The supplied transformed KIBA scale also uses higher scores for stronger affinity; it is not pKd.

Collaborative filtering uses patterns in the observed interaction table. Drugs play the recommender-system role of users, proteins of items, and affinities of ratings. A missing score is predicted from shared interaction patterns, rather than looked up from that pair's held-out label.

| Model | Main mechanism |
|---|---|
| [Global Mean](global_mean.ipynb) | Predict the training-average score for every pair |
| [Protein CF](utils/protein_cf.py) | Predict using positively correlated neighboring proteins and the query drug's observed scores |
| [Drug CF](utils/drug_cf.py) | Use neighboring drugs by exchanging roles in the protein-CF implementation |
| [Biased MF](utils/biased_mf.py) | Learn drug/target offsets and interacting latent vectors jointly from observed training pairs |

Biased MF is a model-based alternative to neighbor lookup. It can share information across the table without selecting neighbors at inference time. It is not guaranteed to outperform KNN; that comparison remains a validation-model-selection task.

**SMILES and amino-acid sequences are exact identifiers here.** They are not converted into fingerprints, molecular graphs, sequence alignments or pretrained embeddings. `predict_biased_mf(model, pairs)` reads only the two ID columns and returns a finite NumPy vector of shape `(len(pairs),)` in input row order. It ignores an `affinity` column even if one is present.

The task is **warm-start**: a queried pair is held out, but both individual IDs occur in training. Unknown drug/target IDs raise `ValueError`; this implementation does not infer a new entity's vector from its chemical or biological features.

## Datasets, preparation and splits

### Sources and identity conventions

The project brief provides DeepDTA and Kaggle benchmark sources. Locally, the pipeline starts from the three-column supplied CSVs; fitting does not download data.

| Dataset | Supplied source | Correct cleaned Task 2 source |
|---|---|---|
| DAVIS | [`../data/davis_all.csv`](../data/davis_all.csv) | [`data/davis_clean_task2.csv`](data/davis_clean_task2.csv) |
| KIBA | [`../data/kiba_all.csv`](../data/kiba_all.csv) | [`../data/kiba_clean.csv`](../data/kiba_clean.csv) |

[data_preparation.ipynb](data_preparation.ipynb) documents filtering, pair aggregation and the fixed split checks. The MF runner itself loads only saved train/validation CSVs.

| Representation | DAVIS drugs / targets / rows | KIBA drugs / targets / rows |
|---|---|---|
| Original benchmark counts in the brief | 68 / 442 / 30,056 | 2,111 / 229 / 118,254 |
| Exact strings in local supplied CSVs | 68 / 379 / 30,056 | 2,068 / 229 / 118,254 |
| Deduplicated string-ID pairs | **68 / 379 / 25,772** | **2,068 / 229 / 117,657** |

Original benchmark IDs are not columns in these local files. Different original records can have identical SMILES or sequences. Grouping by the two strings averages repeated affinities: 4,284 excess DAVIS rows and 597 excess KIBA rows were combined. The recorded cleaning runs removed no invalid-score or blank-ID rows. The tables have three columns, so their cleaned shapes are `(25772, 3)` and `(117657, 3)`.

**Important provenance distinction:** the root [Task 1 notebook](../dataset-analysis.ipynb) applied another logarithmic conversion to the supplied DAVIS values. Its [`../data/davis_clean.csv`](../data/davis_clean.csv) and [summary](../outputs/dataset_summary.csv) are not the Task 2 DAVIS label source. Task 2 correctly preserves already-pKd values in `data/davis_clean_task2.csv`. The older files remain teammate/project history and were not altered during MF consolidation.

### Input columns and scales

| Column | Meaning | Use |
|---|---|---|
| `compound_iso_smiles` | Drug identifier string | Lookup drug parameter row |
| `target_sequence` | Protein identifier string | Lookup target parameter row |
| `affinity` | Measured score | Training label or separate evaluation label |

For background, pKd is:

$$\mathrm{p}K_d=-\log_{10}(K_d\text{ in mol/L})=9-\log_{10}(K_d\text{ in nM}).$$

Illustratively, 100 nM corresponds to pKd 7. **This conversion is not performed by Task 2 MF:** DAVIS already contains pKd. Its cleaned range is 5.0–10.6197887583. KIBA retains the supplied transformed score range 0.0–17.200179498; no additional transformation or normalization is applied.

### Missing entries are not zero

| Dataset | Matrix | Possible positions | Observed cleaned pairs | Cleaned coverage | Training coverage of all positions |
|---|---|---:|---:|---:|---:|
| DAVIS | 68 × 379 | 25,772 | 25,772 | 100% | 69.9984% |
| KIBA | 2,068 × 229 | 473,572 | 117,657 | 24.8446% | 17.3910% |

An unmeasured KIBA pair has no supplied label. Filling it with zero would invent supervision; genuine zero scores, including the recorded KIBA zero, remain valid observations. DAVIS is complete at the cleaned string-ID level, but training is incomplete because validation/test pairs are withheld.

### Fixed partitions

| Dataset | Split | File | Rows | Share of cleaned observed pairs |
|---|---|---|---:|---:|
| DAVIS | Train | [davis_train.csv](data/splits/davis_train.csv) | 18,040 | 69.9984% |
| DAVIS | Validation | [davis_val.csv](data/splits/davis_val.csv) | 3,865 | 14.9969% |
| DAVIS | Test | `data/splits/davis_test.csv` | 3,867 | 15.0047% |
| KIBA | Train | [kiba_train.csv](data/splits/kiba_train.csv) | 82,359 | 69.9992% |
| KIBA | Validation | [kiba_val.csv](data/splits/kiba_val.csv) | 17,648 | 14.9995% |
| KIBA | Test | `data/splits/kiba_test.csv` | 17,650 | 15.0012% |

Test sizes are from the existing preparation audit; no test content is needed to inspect or verify MF. The coverage-aware split uses seed 42, reserves observations covering every drug and protein in training, then fills the requested approximate 70/15/15 proportions. Integer rounding explains the small percentage differences.

The runner verifies train/validation uniqueness, disjoint pairs, finite labels and validation-ID coverage. Earlier preparation checks also covered all three partitions. The campaign's checks and exact train/validation hashes remain in `results/biased_mf/data_integrity.csv`.

### Training-only mapping and leakage prevention

The core constructs sorted unique Unicode ID arrays from training:

```python
drugs = np.sort(training[DRUG].unique()).astype(str)
targets = np.sort(training[PROTEIN].unique()).astype(str)
```

Positions in these arrays define integer indices. `pd.Index(...).get_indexer(...)` maps query strings; unknown IDs return `-1` and are rejected before indexing.

The fixed mean is $\mu=N^{-1}\sum_{(i,j)\in\Omega}y_{ij}$ over training only:

- DAVIS: **5.391476305730676**.
- KIBA: **11.721006434637527**.

Only training labels produce gradients. Validation labels select epochs and configurations, never mappings, means or updates. Prediction accepts ID-only frames. Duplicate-pair aggregation occurs before splitting, avoiding repeated pairs crossing partitions. No final-test loader exists in the MF runner.

Validation selection can change the returned epoch, so fitting is not entirely independent of validation labels. The tested leakage property is more precise: **holding training data and update count fixed, changing validation labels leaves every parameter update identical**.

## Prediction equation and matrix dimensions

$$\hat y_{ij}=\mu+b_i+c_j+\mathbf p_i^\top\mathbf q_j
=\mu+b_i+c_j+\sum_{f=1}^{F}P_{if}Q_{jf}.$$

| Symbol | Meaning | Status |
|---|---|---|
| $y_{ij}$ | Actual measured affinity | Fixed supplied label |
| $\hat y_{ij}$ | Predicted affinity | Calculated |
| $\mu$ | Training global mean | Calculated once, then fixed |
| $b_i$ | Drug-specific score offset | Learned, initialized at zero |
| $c_j$ | Protein-specific score offset | Learned, initialized at zero |
| $\mathbf p_i$ | Drug latent vector | Learned, small seeded random initialization |
| $\mathbf q_j$ | Protein latent vector | Learned, small seeded random initialization |

The mean establishes the score scale; biases describe broad tendencies toward higher/lower scores. The dot product adds pair-specific compatibility beyond those tendencies. Latent coordinates are not explicitly labeled biological properties, and equivalent factor representations can yield the same predictions.

Let D/T count training drugs/targets, F count factors, N/V count training/validation observations, and B be the current batch length.

| Object | Definition / mathematical shape | Actual representation and use |
|---|---|---|
| $R$ | Partially observed affinities, D × T | Conceptual; not allocated by MF; unknown entries are unspecified |
| $\Omega$ | Set of N observed training pairs | Fixed integer coordinate arrays `d`, `t`, each `(N,)` |
| $M$ | Binary training mask, D × T | Conceptual only; 1 marks an observed training pair, not strong binding |
| $P$ | Drug factor matrix, D × F | `drug_factors`, initialized Normal(0, 0.01²), learned |
| $Q$ | Target factor matrix, T × F | `target_factors`, same initialization distribution, learned |
| $b,c$ | Bias column vectors, D × 1 and T × 1 | One-dimensional NumPy `drug_bias`, `target_bias`; zero initially, learned |
| $\hat R$ | Complete known-ID prediction table, D × T | Calculated conceptually; only requested pairs are evaluated |
| `y`, `vy` | Fixed labels, `(N,)`, `(V,)` | Float64; `vy` is for evaluation only |
| `vd`, `vt` | Validation coordinates, `(V,)` | Calculated from training mappings |
| `order`, `rows` | Permutation `(N,)`, batch indices `(B,)` | New shuffled order per epoch; final batch may be shorter |
| Local `p`, `q` | Gathered factor rows, `(B,F)` | Pre-update factor values for batch gradients |
| `error` | Normalized residuals, `(B,)` | Actual code stores $(\hat y-y)/B$ |
| `gradients` | Four arrays matching b, c, P, Q | Recalculated data-plus-L2 derivatives per batch |
| `first`, `second` | Two sets of parameter-shaped arrays | Adam moments, zero initially, updated per batch; not saved |
| `step` | Scalar update counter | Starts at zero, increases per batch, not reset per epoch |
| `best` | Copies of the four parameter arrays | Replaced on lower validation MSE; restored after stopping |
| `history` | `(epochs_run+1,4)` | Epoch, training MSE, validation MSE, regularized training objective |

### Dimensions of the selected F=128 models

| Quantity | DAVIS | KIBA |
|---|---|---|
| D, T, N, V | 68, 379, 18,040, 3,865 | 2,068, 229, 82,359, 17,648 |
| $R$, $M$, $\hat R$ | 68 × 379 | 2,068 × 229 |
| P | `(68,128)` | `(2068,128)` |
| Q | `(379,128)` | `(229,128)` |
| $Q^\top$ | 128 × 379 | 128 × 229 |
| b, c | `(68,)`, `(379,)` | `(2068,)`, `(229,)` |
| Recorded history | `(85,4)`, epochs 0–84 | `(45,4)`, epochs 0–44 |
| Learned scalar count, $(D+T)(F+1)$ | **57,663** | **296,313** |

Each gradient and each Adam moment has the same shape as its parameter. Factorization does not necessarily compress the table at the chosen rank: these parameter counts exceed the training observation counts, and DAVIS's interaction matrix rank is bounded by 68 even at F=128. Overparameterization, optimization and regularization all influence the result.

### Full matrix equation

Using all-ones column vectors $\mathbf1_D$ and $\mathbf1_T$:

$$\hat R=\mu\mathbf1_D\mathbf1_T^\top+b\mathbf1_T^\top+\mathbf1_Dc^\top+PQ^\top.$$

Every term is D × T. The outer products broadcast the mean and biases. Transposition changes Q from T × F to F × T; multiplying P by $Q^\top$ sums matching latent coordinates, rather than multiplying complete matrices elementwise. A single dot product is the sum of coordinate-wise products for one drug/target row pair.

MF always uses drug rows and protein columns for both datasets. Drug CF exchanges roles to reuse a neighbor-similarity implementation; that transposition is unnecessary for MF, which learns both sets of factors jointly. Different dimensions and sparsity only change array sizes and which observed pairs contribute gradients.

### Illustrative two-drug, three-target example

Let $\mu=6$, $b=(0.3,-0.2)^\top$, $c=(0.1,-0.1,0.2)^\top$, and:

$$P=\begin{bmatrix}0.2&0.4\\-0.1&0.3\end{bmatrix},\qquad
Q=\begin{bmatrix}0.5&0.2\\0.1&-0.2\\-0.3&0.4\end{bmatrix}.$$

Drug 1/target 1 has interaction $(0.2)(0.5)+(0.4)(0.2)=0.18$, hence prediction $6+0.3+0.1+0.18=6.58$. The full calculated prediction table is:

$$\hat R=\begin{bmatrix}6.58&6.14&6.60\\5.91&5.63&6.15\end{bmatrix}.$$

No actual labels are needed to calculate those predictions. In the code, indexed bias values plus `np.einsum("ij,ij->i", p, q)` compute one dot product per requested row. Queries are processed in chunks of at most 8,192 to limit temporary arrays.

## Squared error, objective and regularization

The code uses residual $e_{ij}=\hat y_{ij}-y_{ij}$. Positive means overprediction; negative means underprediction. Squaring prevents cancellation and penalizes large errors more heavily. For a set of n observations:

$$\mathrm{MSE}=\frac1n\sum_{\ell=1}^{n}(\hat y_\ell-y_\ell)^2.$$

Illustratively, residuals −1, 0 and 2 give MSE $5/3$. MSE is sensitive to outliers and has squared-score units; numeric DAVIS and KIBA MSEs are not directly comparable measures of dataset difficulty.

Define $S=\|b\|_2^2+\|c\|_2^2+\|P\|_F^2+\|Q\|_F^2$, the sum of every learned coefficient's square. The unchanged objective is:

$$L=\frac{\sum_{(i,j)\in\Omega}e_{ij}^2+\lambda S}{2N}
=\frac12\mathrm{MSE}_{\rm train}+\frac{\lambda}{2N}S.$$

- N is the number of **observed training pairs**, not D×T or batch size.
- The fixed global mean is not regularized; both bias vectors and both factor matrices are.
- The penalty is one global parameter sum, not a per-observation repetition weighted by entity frequency.
- The factor 1/2 simplifies derivatives. `train_objective` is not MSE.
- Validation selection uses unregularized validation MSE.
- This is not `MSE + lambda * S`; the numerical convention matters. Relative to the mean data term, shrinkage scales with lambda/N, so the same lambda does not imply identical relative regularization across different dataset sizes.

L2 discourages large coefficients. Too little can permit training-specific patterns; too much can underfit. It does not impose sparsity or guarantee improved validation performance.

In the illustrative matrix above, parameter squared norms sum to $S=0.13+0.06+0.30+0.59=1.08$. Suppose two observations have actual/predicted scores 6.80/6.58 and 6.00/6.15. The squared-error sum is $0.0484+0.0225=0.0709$. With N=2 and illustrative lambda=0.1:

$$\mathrm{MSE}_{\rm train}=0.03545,\qquad L=(0.0709+0.1\times1.08)/4=0.044725.$$

**Underfitting** can produce high training and validation error through insufficient capacity, excessive shrinkage or incomplete optimization. **Overfitting** is supported when further training improves training error but worsens held-out performance. A training–validation gap alone is not proof, nor is the smallest gap automatically best.

## Gradients and Adam

A coefficient is an adjustable bias or latent-factor element. Its gradient measures the local change in loss as that coefficient changes. Ordinary gradient descent subtracts the gradient: a positive gradient decreases the parameter, a negative one increases it. A zero derivative does not imply a global optimum in this nonconvex model.

The chain rule for one observation gives:

$$\frac{\partial(\tfrac12 e_{ij}^2)}{\partial\theta}=e_{ij}\frac{\partial\hat y_{ij}}{\partial\theta}.$$

Bias derivatives of the prediction are 1; factor derivatives are the other entity's coordinate: $\partial\hat y/\partial P_{if}=Q_{jf}$ and $\partial\hat y/\partial Q_{jf}=P_{if}$.

### Exact mini-batch normalization

For batch $\mathcal B$ with B rows, drug/target indices $d_\ell,t_\ell$ and residual $e_\ell$:

$$g_{b_i}=\frac1B\sum_{\ell:d_\ell=i}e_\ell+\frac\lambda N b_i,\qquad
g_{c_j}=\frac1B\sum_{\ell:t_\ell=j}e_\ell+\frac\lambda N c_j,$$

$$g_{P_{if}}=\frac1B\sum_{\ell:d_\ell=i}e_\ell Q_{t_\ell f}+\frac\lambda N P_{if},\qquad
g_{Q_{jf}}=\frac1B\sum_{\ell:t_\ell=j}e_\ell P_{d_\ell f}+\frac\lambda N Q_{jf}.$$

The sums are over the current batch; the L2 term always uses the full N. All parameter rows receive regularization, even if absent from the batch. At fixed parameters, the size-weighted average of batch gradients equals the full gradient:

$$\sum_r\frac{|\mathcal B_r|}{N}\nabla L_{\mathcal B_r}=\nabla L.$$

This includes a short final batch. Changing B does not redefine the L2 coefficient, but it changes update count, gradient noise and Adam's trajectory, so identical fitted models are not expected across batch sizes.

### Repeated indices and simultaneous gradients

Multiple pairs in a batch may share a drug or target. All contributions must accumulate. For example, drug indices `[0,0,1]` with residuals `[2,-1,3]` produce unregularized bias gradients `[1/3,1]`, not just the last contribution for drug 0. NumPy fancy-index `+=` is unsuitable for reliably accumulating repeated indices; the core uses `np.bincount`.

Actual source excerpt:

```python
error = (_predict_indices(model, drug, target) - scores) / len(scores)
nd, nt = len(model["drugs"]), len(model["targets"])
gradients = {key: (regularization / n_training) * model[key] for key in PARAMETERS}
gradients["drug_bias"] += np.bincount(drug, weights=error, minlength=nd)
gradients["target_bias"] += np.bincount(target, weights=error, minlength=nt)
p, q = model["drug_factors"][drug], model["target_factors"][target]
```

This computes normalized residuals, initializes full L2 gradients, accumulates biases and gathers both factor sets before updating either. The following coordinate loop accumulates `error * q[:, factor]` by drug and `error * p[:, factor]` by target. All four gradients therefore use the same pre-update parameter state.

### Adam update

Unlike plain SGD, Adam maintains moving averages of the gradient and squared gradient for each coefficient:

$$m_s=0.9m_{s-1}+0.1g_s,\quad v_s=0.999v_{s-1}+0.001g_s^2,$$
$$\hat m_s=m_s/(1-0.9^s),\quad\hat v_s=v_s/(1-0.999^s),$$
$$\theta_s=\theta_{s-1}-\alpha\frac{\hat m_s}{\sqrt{\hat v_s}+10^{-8}}.$$

Moments start at zero, so bias correction compensates for that initialization. The second moment is a moving average of squared gradients, not centered variance. The step counter counts batches across epochs. Momentum can move a parameter even when its current gradient is zero.

L2 is already included in $g_s$: this is **Adam with coupled L2**, not AdamW. Adam changes coefficients, never observed affinity labels. Adaptive scaling may help with differing gradient magnitudes and entity frequencies, but the base learning rate still matters. No claim that Adam outperforms SGD is supported here because SGD was not evaluated.

Illustratively, for $\theta_0=0.2$, $g_1=-0.1$, alpha=0.01: $m_1=-0.01$, $v_1=0.00001$, corrected moments are −0.1 and 0.01, so $\theta_1\approx0.21$. Plain SGD at that rate would produce 0.201; this illustrates different rules, not relative benchmark quality.

## Training and early stopping

1. Load and validate the fixed training and validation tables.
2. Build mappings and mean from training. Initialize biases to zero and factors with seeded Normal(0, 0.01²) draws.
3. Initialize Adam moments to zero; evaluate epoch zero as an eligible checkpoint.
4. Shuffle training indices using the local NumPy generator. Process each observation once per epoch in batches up to 2,048.
5. Predict each batch, calculate all gradients from the pre-update state, increment the step counter and apply Adam.
6. After each epoch, calculate full training MSE, validation MSE and the regularized training objective. These are end-of-epoch metrics, not averaged within-epoch losses.
7. A strictly lower validation MSE replaces the best parameter copies and resets patience. Equal MSE is not an improvement; there is no `min_delta`.
8. Stop after 10 consecutive non-improvements or at the epoch cap. Restore the best parameters while retaining the entire attempted history.
9. Evaluate the restored state using shared metrics; save parameters and metadata as numeric/Unicode `.npz` arrays.

DAVIS has 9 batches per epoch, with last batch 1,656 rows. KIBA has 41, with last batch 439. Epoch counts therefore do not imply comparable update counts across datasets.

The final DAVIS model stops at epoch 84 but restores epoch **74**; KIBA stops at 44 but restores **34**. Histories include later rejected epochs, not only the chosen state. Best parameters are copied in memory during fitting; the model is not saved to disk at every epoch. There is no refit on training plus validation.

Seed 42 controls initialization and shuffling. Changing F consumes a different number of random draws, so same seed labels do not guarantee identical shuffle sequences across dimensions. Within the same environment/settings/data order, determinism is tested. Bitwise reproducibility across different NumPy/BLAS environments is not guaranteed.

Inputs, predictions, gradients, parameters and epoch metrics have finite-value checks. Unknown IDs fail rather than wrapping negative indices. During fitting, overflow/invalid operations raise informative exceptions with epoch/configuration. A numerical failure is recorded, not silently repaired by clipping. There were no numerical failures in the campaign.

## Scientific experimental campaign

### Stage 1 — capacity

F={8,16,32,64,128}, lambda={0,0.01}, alpha=0.005, seed=42, cap=100, patience=10: **20 conditions** across two datasets. Unregularized fits are scientific comparisons, not an automatic final-rank decision.

Training and best-validation MSE decrease with every tested increase in F for both penalties/datasets. At lambda=0.01, DAVIS validation MSE improves from 0.323905 at F=8 to 0.268317 at F=128; KIBA improves from 0.244911 to 0.198213. The experiment does not show an inverted-U best-validation curve in this range.

### Stage 2 — overfitting analysis

All 20 capacity trajectories improve training MSE after the selected epoch while ending with worse validation MSE. The descriptive flag requires both changes to exceed 1e-6, not just a positive gap. Best epochs move earlier as F increases. Nevertheless, the larger models have lower best-validation MSE even as their gaps grow. Smaller gaps are therefore not the selection rule.

See `results/biased_mf/STAGE2_INTERPRETATION.md` and `results/biased_mf/stage2_curve_diagnostics.csv`.

### Stage 3 — joint F–lambda interaction

The five F values are crossed with lambda={0,0.01,0.1,1,10}, at alpha=0.005 and seed 42: **25 conditions per dataset**, including verified reuse of Stage 1. Full-precision metrics are in `results/biased_mf/stage3_interaction.csv`.

- DAVIS prefers lambda=1 for F=16–128, but F=8 prefers 10 within the 100-epoch budget.
- KIBA prefers lambda=1 at every F, but its preferred F changes from 128 to 32 when lambda=10.
- Stronger regularization is not uniformly beneficial. At large F, lambda=10 raises both errors relative to lambda=1.
- Four strong-penalty core conditions remain improving at the 100-epoch cap. They are flagged, not described as converged; the core study retains its specified common budget.

*Local/generated figure: `results/biased_mf/figures/davis_interaction_heatmaps.png` — DAVIS F–lambda heatmaps.*
*Local/generated figure: `results/biased_mf/figures/kiba_interaction_heatmaps.png` — KIBA F–lambda heatmaps.*

Panels have separately labeled scales; actual values are unsmoothed. Best-lambda and best-F tables are retained, along with curves for no, moderate and strong regularization. Historical lambda=100 outcomes remain compact supplementary evidence rather than new campaign fits.

### Stage 4 — learning-rate refinement

Choose the two best F–lambda pairs per dataset by Stage 3 validation MSE. Test alpha={0.001,0.003,0.005,0.01}, holding the objective, batch size and seed fixed. The refined pairs were DAVIS (128,1)/(64,1), and KIBA (128,1)/(128,0.1).

**Predefined extension rule:** every rate begins with cap=100. If it reaches the cap with a new minimum in its last ten epochs, replay from the same seed with cap=300 and unchanged patience. Preserve both physical fits and verify exact equality of the first 101 history rows. Apply the same rule to robustness seeds; never extend beyond 300.

DAVIS F=64/lambda=1/alpha=0.001 triggered it: best epoch 100 became 105, stopping at 115. Both initial and extended update costs are retained. All final refined comparisons stopped through patience, with no numerical failures. Alpha=0.001 gave better validation MSE in every refined family, at the cost of more updates. This does not imply every learning rate eventually reaches the same solution.

### Stage 5 — seed robustness

Freeze the leading two **full configurations** per dataset, then assess seeds 42, 123 and 2026. Reuse verified seed-42 results. Final configuration selection uses the mean validation MSE across all three; report sample SD (`ddof=1`). This is additional validation selection, not independent testing.

| Candidate | Mean validation MSE ± sample SD |
|---|---|
| DAVIS F64, lambda=1, alpha=.001 | 0.261293 ± 0.001220 |
| **DAVIS F128, lambda=1, alpha=.001** | **0.258847 ± 0.002471** |
| **KIBA F128, lambda=1, alpha=.001** | **0.189855 ± 0.001323** |
| KIBA F128, lambda=1, alpha=.003 | 0.190666 ± 0.001393 |

Two more DAVIS F64 runs triggered extensions. Seed 123 selected epoch 112; seed 2026 found no better epoch and retained 97, stopping at 107. Extension is not guaranteed to help. Overall there were **70 base conditions plus 3 extension replays = 73 physical fits**.

Three seeds are limited evidence, not statistical proof. DAVIS F128 has lower mean MSE but greater variability and lower mean Hit@K than F64. The primary criterion, not dominance on every metric, determines selection. The selected checkpoint uses seed 42 even though DAVIS seed 123 achieved a lower MSE.

## Evaluation metrics and candidate counts

All metrics come from the unchanged [shared helpers](utils/evaluation.py).

### MSE

MSE is the mean squared score error; lower is better. Every observation receives equal weight, so a ligand with more measurements contributes more terms. MSE checks score calibration as well as ranking. The shared MSE helper computes the mean directly and does not perform the shape/finiteness validation used by `_scores`; the MF pipeline supplies valid aligned arrays.

### Concordance Index

Let $\mathcal C=\{(a,b):a<b, y_a\ne y_b\}$. For each comparable observation pair, award 1 for matching actual/predicted order, 1/2 for exactly tied predictions, and 0 for reversal:

$$\mathrm{CI}=\frac{\sum_{(a,b)\in\mathcal C}\left[\mathbf1\{(y_a-y_b)(\hat y_a-\hat y_b)>0\}+\tfrac12\mathbf1\{\hat y_a=\hat y_b\}\right]}{|\mathcal C|}.$$

Equal actual scores are excluded. No comparable pairs returns NaN. The implementation compares **globally across all observations**, not separately within each drug, and uses a rank/prefix-sum algorithm in O(n log n). Exact equality defines ties.

Illustratively, actual `[1,2,2,3]` and predictions `[.2,.4,.4,.3]` give five comparable pairs: three correct, two reversed, hence CI=0.6. With actual `[1,2]` and tied predictions `[.5,.5]`, CI=0.5. Constant predictions have CI=0.5 when actual comparable pairs exist. Good CI need not imply well-calibrated numeric scores or good within-ligand rankings.

### Hit@1 and Hit@5

For each ligand i, $C_i$ is its set of observed validation targets and $A_i$ is the set tied for highest actual score. Rank predictions descending, breaking prediction ties by the target ID string ascending. With predicted top-$\min(K,|C_i|)$ set $T_i^K$:

$$\mathrm{Hit}@K=\frac1{|\mathcal L|}\sum_{i\in\mathcal L}\mathbf1\{T_i^K\cap A_i\ne\varnothing\}.$$

All actual-best ties count; actual scores never break prediction ties. Each eligible ligand gets one vote, regardless of candidate count. The helper checks unique pairs and returns both hit rate and eligible count. Empty candidates produce `(NaN,0)` through dictionary fields.

Illustratively, actual-best targets are B and E, and the predicted ranking is C,A,D,E,B,F. Hit@1=0 and Hit@5=1. If a ligand has at most five candidates, Hit@5 is automatically 1; a singleton also always hits at K=1.

| Dataset | Eligible ligands | Candidate min / Q1 / median / Q3 / max | Singletons | At most 5 candidates | More than 5 |
|---|---:|---|---:|---:|---:|
| DAVIS | 68 | 45 / 53 / 57.5 / 60 / 79 | 0 | 0 | 68 |
| KIBA | 1,967 | 1 / 2 / 6 / 15 / 32 | 248 | 959 | 1,008 |

These are measured held-out candidate sets, not all proteins. The same helper additionally reports Hit@5 after restricting to ligands with more than five candidates. The final model has DAVIS restricted Hit@5=0.720588 and KIBA restricted Hit@5=**0.906746** (overall KIBA=0.952211). The supplementary metric is not a selection criterion. Exact counts are in `results/biased_mf/candidate_counts.csv`.

## Final selection and interpretation

The final configuration is F=128, lambda=1, alpha=0.001, batch=2048, seed=42, patience=10. Both selected seed-42 fits used cap=100 and stopped before it.

| Dataset | Versus earlier F64/alpha=.005/lambda=.01 | Versus earlier F64/alpha=.005/lambda=1 |
|---|---|---|
| DAVIS | **4.3562% lower validation MSE** | **2.8421% lower** |
| KIBA | **7.8456% lower validation MSE** | **5.2604% lower** |

Training–validation gaps at the published epochs are **0.2336224641** (DAVIS) and **0.1655133115** (KIBA). Their size did not determine selection. Full historical comparison values are in `results/biased_mf/reference_comparison.csv`.

The campaign improves MSE and scientific understanding, but not every metric. DAVIS Hit@1 is below the original F64/lambda=.01 reference, and Hit@5 is below the earlier F64/lambda=1 sweep winner. KIBA's candidate-set distribution also makes raw Hit@5 easier than DAVIS's. Interpret these trade-offs rather than claiming a universally superior model.

The selected checkpoints are `saved_models/davis_biased_mf.npz` and `saved_models/kiba_biased_mf.npz`. They are the campaign F128 models, **not the obsolete F64/lambda=.01 checkpoints**. They were promoted byte-for-byte, preserving all full-precision values and original provenance metadata.

## Code and artifact guide

The tree below describes the **complete local workspace after experimentation**.
Only the three Python files, this README, `REPORT.md`, `PROTOCOL.md` and the
repository `.gitignore` are part of this minimal addition to GitHub. Existing
shared datasets/evaluation code are already tracked in the project.

```text
task2/
├── utils/biased_mf.py          # unchanged model, gradients, Adam and serialization
├── run_biased_mf.py            # official staged campaign + read-only verifier
├── biased_mf.ipynb             # executed reporting notebook; no training cells
├── tests/test_biased_mf.py     # original 18 correctness tests
├── README.md
├── saved_models/{davis,kiba}_biased_mf.npz
└── results/biased_mf/
    ├── REPORT.md, PROTOCOL.md, STAGE2_INTERPRETATION.md
    ├── stage*.csv, final_selection.csv, reference_comparison.csv
    ├── all_physical_runs.csv, all_training_histories.csv
    ├── candidate_counts.csv, candidate_summary.csv
    ├── figures/
    ├── manifest.json, verification.json, consolidation_checks.json
    ├── RETENTION.md, LOCAL_REVIEW.md
    └── provenance/
        ├── original_manifest.json, original_sources.zip
        ├── historical_results.csv
        └── consolidation.json
```

### `utils/biased_mf.py`

| Function | Inputs / output / responsibility |
|---|---|
| `fit_biased_mf(training, validation, ...)` | Labeled DataFrames and hyperparameters → dictionary containing restored-best parameters, mappings, metadata and history |
| `predict_biased_mf(model, pairs)` | Model plus ID columns → finite score vector; unknown IDs fail; empty valid input yields `(0,)` |
| `save_biased_mf(model, path)` | Validated dictionary → compressed `.npz`; numeric/Unicode fields only; caller supplies existing parent |
| `load_biased_mf(path)` | Loads with `allow_pickle=False`, copies arrays and validates mappings/shapes/finiteness |
| `_check_pairs`, `_indices` | Validate frames/labels and construct query indices |
| `_predict_indices` | Chunked rowwise factor/bias prediction |
| `_objective`, `_batch_gradients` | Exact observed-only regularized objective and accumulated derivatives |
| `_adam_update` | In-place coefficient/moment updates, with finite-value guards |
| `_validate_model` | Format version, unique nonblank Unicode IDs, factor/bias dimensions, finite floating arrays and mean |

The model is a dictionary, not an estimator class. Public fit defaults remain F=16/alpha=.01/lambda=.001 for API compatibility; the official runner passes campaign settings explicitly. Loaded scalar fields are zero-dimensional arrays; use `int(...)`, `float(...)` or `.item()` as appropriate. Direct API callers must ensure split disjointness: that cross-frame check belongs to the runner.

### `run_biased_mf.py`

This is the only active experiment runner. It imports the model and shared metrics, not another historical runner.

- `load_selection_splits`, `candidate_distribution`, `evaluate_validation`: the original helpers, moved without semantic changes. Prediction gets ID-only frames; labels go separately to metrics.
- `config`, `run_id`, `current_fingerprints`: reproducible configuration identifiers and SHA-256 provenance.
- `prepare`: creates a fresh campaign directory, plans and manifest; never overwrites published evidence.
- `Campaign.trial`: fit or verify/reuse one full run, retaining metrics, history, state and prediction hashes in fresh reproductions.
- `Campaign.extended_trial`: applies the common 100→300 rule and checks exact history-prefix equality.
- `stage_capacity`, `stage_interaction`, `stage_rates`, `stage_seeds`: the scientific stages, selection rules and figures.
- `verify_all` (CLI `summarize`): validates full reproduction-run states and builds aggregate tables once.
- `verify_published` (CLI `verify`): read-only verification of current source/data hashes, original lineage, all retained histories, seed summaries and the two selected checkpoints; **does not require the 73-run archive or fit any model**.
- `historical_rows`: reads the compact 40-condition comparison table if available; otherwise returns an empty, correctly structured table. Fresh code-only runs fit the same planned grid without requiring any historical artifact. Missing comparisons are recorded as unavailable, not as successful checks. It never interprets the selected F128 models as historical F64 checkpoints.
- `curve_plot`, `peer_evidence`, `sort_results`: visualization, descriptive diagnostics and deterministic campaign ordering.
- `run_search(root, grid, ...)`, `select_best`, `_curve`: a small **explicit-grid programmatic compatibility API** used by the original synthetic integration test. There is no default old grid, obsolete baseline cache or CLI `baseline/search` workflow. Existing checkpoint/result paths are protected. It calls the same model and evaluation helpers, not a second optimizer.

The active consolidated manifest records the refactored source hash. The archived original manifest/source bundle retains what actually produced the trained states; consolidation does not pretend that training ran under a source file created later.

### `biased_mf.ipynb`

The reporting notebook reads consolidated CSVs and figures, displays each stage and its scientific interpretation, and shows final metrics/robustness. It contains **no fit/search calls** and no test-data loader. Its saved outputs are executed. Opening it does not launch the 73-fit campaign. Running its reporting cells only reads evidence; saving the notebook changes notebook outputs, as usual.

### Shared evaluation and test files

`utils/evaluation.py` supplies `_scores`, MSE, global CI and Hit@K. It is unchanged and shared with Jiaxin's models. `tests/test_biased_mf.py` retains its three original test classes and all 18 test methods unchanged; see the next section.

### Retained evidence and checkpoint contents

- `stage1_capacity.csv`: 20 capacity conditions, metrics, parameter counts and diagnostics.
- `stage2_curve_diagnostics.csv`: capacity curve evidence, also interpreted before the interaction study.
- `stage3_interaction.csv`: all 50 F–lambda conditions; `stage3_best_*` tables retain conditional winners.
- `stage4_candidates.csv`, `stage4_plan.csv`, `stage4_rates.csv`: frozen pair shortlist, requests and comparable rate outcomes/update counts.
- `stage5_candidates_frozen.csv`, `stage5_plan.csv`, `stage5_seed_results.csv`, `stage5_robustness_summary.csv`: frozen full configurations, all seed metrics and sample mean/SD.
- `all_physical_runs.csv`: 73 physical fits, including extension prefixes and source/prediction/checkpoint hashes.
- `all_training_histories.csv`: 3,275 unsmoothed epoch rows, preserving complete learning curves without redundant per-run CSV copies.
- `final_selection.csv`: full-precision seed-42 metrics, selection metadata and official checkpoint paths.
- Local figures: capacity, F–lambda heatmaps, regularization curves, learning-rate curves and seed variation; the report identifies their generated paths.
- Historical comparisons, candidate counts, tests/verification records, protocol and hashes remain compact supporting evidence.

Checkpoints store global mean, ordered ID mappings, both biases and factor matrices, hyperparameters, seed, score-scale information, best/stop epochs, history, validation metrics, original campaign provenance and three-seed selection metadata. They are **inference checkpoints**, not resumable Adam-state snapshots: moments, optimizer counter, RNG state and raw label tables are absent.

## Correctness tests and verification

All **18 original tests** are retained; they are run again for consolidation. Their small synthetic fits are software verification, not final held-out scientific evaluation.

| Class | Checks |
|---|---|
| `GradientTests` (3) | Central finite differences for every bias/factor element with repeated IDs; weighted batch-gradient/L2 invariance; analytical first Adam update |
| `ModelTests` (12) | Shape/finiteness/empty queries; prediction ignores labels; validation labels leave fixed-length update trajectories unchanged; validation cannot expand mappings; determinism/input preservation; pickle-free round trip; synthetic loss reduction; best-epoch restoration/patience; invalid IDs; non-finite scores/malformed models; invalid hyperparameters; deliberate overflow rejection |
| `ExperimentTests` (3) | Candidate distributions and restricted Hit@5; validation-MSE selection/ties; actual tiny search with a train/validation-only CSV allowlist |

The gradient fixture has four drugs, three targets, two factors and seven observations, with repeated indices and one absent drug. Central differences use epsilon=1e-6 and lambda=.37; a four-observation batch still uses full N=7 in its regularizer. Maximum recorded errors:

| Parameter | B=7 | B=4 |
|---|---:|---:|
| Drug biases | 3.990e-11 | 6.686e-11 |
| Target biases | 4.605e-11 | 1.059e-10 |
| Drug factors | 7.601e-11 | 5.934e-11 |
| Target factors | 5.245e-11 | 9.245e-11 |

Acceptance is `atol=1e-8, rtol=1e-6`. Batch sizes 1,2,3,4,7 pass full-gradient/L2 checks at `atol=1e-15, rtol=1e-13`; this states the actual tolerance, not an independently measured maximum absolute error. Synthetic best-epoch MSE and objective fall below 20% of their initial values. A controlled validation curve stops at epoch 4 and restores exact epoch-2 parameters.

The original campaign verified all 73 intermediate checkpoint/prediction states. That recorded evidence is preserved, not falsely claimed as a new check after removing intermediate files. Current `--stage verify` independently checks **both final checkpoints**, their exact training/validation prediction hashes, all 73 aggregate histories/objective values/update counts, and the seed-based selection. The historical core/evaluation hashes still match the current byte-identical implementations.

## WSL commands

### Setup and software tests

From the repository root:

```bash
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
mkdir -p /tmp/opencode

python -B -m unittest discover -s task2/tests -p 'test_biased_mf.py' -v
```

The existing environment used Python 3.12.3, NumPy 2.5.4 and pandas 3.1.1. Matplotlib and Jupyter support figures/notebooks; pytest is unnecessary. The tests use `/tmp/opencode` as a temporary parent. `OPENBLAS_NUM_THREADS=1`, set before NumPy imports, matches the recorded runs. `PYTHONDONTWRITEBYTECODE=1` avoids new bytecode caches.

On a fresh machine without `.venv`, first create it with `python3 -m venv .venv`,
activate it, and install `numpy`, `pandas` and `matplotlib` using pip. Jupyter is
optional and needed only when the local reporting notebook is available. No
environment or notebook is bundled in this code-only push.

If the original local artifacts are available, verify/read them with:

```bash
python -B -m task2.run_biased_mf --stage verify
jupyter notebook task2/biased_mf.ipynb
```

These two commands require local artifacts absent from a fresh GitHub checkout.
The verifier gives an actionable error when outputs are missing and never starts
training automatically. For reproduced artifacts, pass their `--output-dir`.

The pandas environment emits a deprecation warning for the existing `float_precision="round_trip"` option. It remains supported and is not a training numerical failure. No new dependencies are required.

### Deliberate full reproduction — not needed to inspect results

**These commands train models.** Use a new output directory; retained local evidence is read-only. Future full-run caches and checkpoints in the reproduction directory are ignored by Git. This is how a code-only checkout generates its own results; it is unnecessary when merely inspecting the numerical findings in the report.

```bash
OUT=task2/results/biased_mf_reproductions/run1
for STAGE in prepare smoke tests capacity interaction rates seeds summarize verify; do
    python -B -m task2.run_biased_mf --stage "$STAGE" --output-dir "$OUT" || break
done
```

`prepare` refuses an existing directory. Later stages reuse only matching full-run states within that new directory. `summarize` creates aggregate histories/inventory; `verify` is read-only. New selected models remain in that output's `checkpoints/`, not automatically promoted over local `saved_models/` files. The historical comparison CSV is optional. No notebook, old runner, old checkpoint, provenance ZIP or previous results are required to train a fresh reproduction.

### Inspect metrics or predict without labels

The following example assumes the original local artifacts are present. After a
fresh reproduction, use its `final_selection.csv` and `checkpoints/` paths instead.

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
from task2.utils.biased_mf import PAIR_KEYS, load_biased_mf, predict_biased_mf

root = Path('task2')
print(pd.read_csv(root / 'results/biased_mf/final_selection.csv').to_string(index=False))
model = load_biased_mf(root / 'saved_models/davis_biased_mf.npz')
pairs = pd.read_csv(root / 'data/splits/davis_val.csv', usecols=PAIR_KEYS, nrows=3)
print('F:', int(model['n_factors']), 'lambda:', float(model['regularization']))
print('best epoch:', int(model['best_epoch']))
print(predict_biased_mf(model, pairs))
PY
```

Substitute `kiba` for the other model. The example uses only identifier columns and does not fit or write anything. Exact known strings are required; a novel ID raises an error. Loading always uses `allow_pickle=False` internally.

## Consolidation, provenance and exclusions

The complete local layout keeps one active runner and reporting notebook, the unchanged model/test code, numerical evidence and two final checkpoints. The minimal GitHub addition contains code and three Markdown documents only, plus `.gitignore`. The earlier baseline/grid/sweep runners and duplicate working outputs were retired after verification.

Before replacement, all MF-owned original files were archived and every archive member hash-checked. A local-only backup is retained at:

```text
.local_artifacts/biased_mf_consolidation/pre_consolidation.tar.gz
```

It contains the original 73-run archive, obsolete checkpoints, notebooks, reports and runners. It is excluded from GitHub and exceeds the ordinary 100 MiB per-file upload limit. Keep it locally for review/recovery; it is not required for inference or compact verification.

**Included in this push:** `.gitignore`, `utils/biased_mf.py`, `run_biased_mf.py`, `tests/test_biased_mf.py`, this README, `results/biased_mf/REPORT.md` and `results/biased_mf/PROTOCOL.md`.

**Not included in this push:** notebooks, all checkpoints, CSVs, figures, logs, provenance ZIP/JSON files, local review/retention documents, `.venv`, caches, backups and reproduction outputs. They remain local; exclusion from staging does not delete them. The final checkpoint sizes are 699,420 and 2,485,559 bytes, but they are excluded for the requested minimal scope, not because they exceed GitHub's limits. Shared datasets and teammate files are already tracked and are not changed by this push.

The original source/data hashes and historical model metadata were not rewritten to pretend refactored code produced past fits. Instead:

- `provenance/original_manifest.json` keeps the original hashes.
- `provenance/original_sources.zip` is a small non-active audit snapshot of the four original source files; archive member bytes match the original hashes.
- `provenance/consolidation.json` records byte-identical model/evaluation files, unchanged ASTs for moved helpers and checkpoint promotion.
- `manifest.json` identifies the current consolidated runner/data and explicitly marks this as retained evidence, not a new training run.

Historical source names inside provenance and checkpoint metadata are intentionally historical identifiers, not active imports or runnable paths. The source ZIP is an audit artifact, not a second installed implementation. See `results/biased_mf/RETENTION.md` for exact retention/recovery policy and `results/biased_mf/LOCAL_REVIEW.md` for the unstaged local change review.

## Limitations and remaining work

- Repeated selection uses one fixed validation split. Early stopping, shortlisting and three-seed mean selection can make the final validation result optimistic.
- Three seeds provide limited evidence. DAVIS ranking metrics vary noticeably over only 68 eligible ligands.
- Four strong-penalty core cells remain cap-limited. Only the two promising families per dataset received rate refinement, not a complete F–lambda–alpha grid.
- ID-only factors cannot naturally generalize to completely unseen drugs/targets. Task 3 requires a different approach, such as transferable molecular/sequence features and genuinely target-disjoint evaluation.
- KIBA's sparse and uneven observations limit support for some entities. Unmeasured pairs are not proven negatives.
- Global CI, measured-candidate Hit@K and scale-dependent MSE measure different things. Lower MSE need not improve every ranking metric.
- Checkpoints support inference, not exact training continuation; no Adam moment/RNG state is saved.

Remaining Task 2 steps:

1. Compare validation results with Jiaxin's Global Mean, Protein CF and Drug CF.
2. Agree on frozen final models using validation evidence.
3. Perform the planned final evaluation without test-driven tuning.
4. Load saved models for three ligand case studies per dataset: recommendations, true/predicted scores, MSE/CI/Hit@K and rank of the true-best target.
5. Prepare the report and presentation, including trade-offs and individual contributions.

This MF workflow has not accessed final-test data. Teammate notebooks contain historical test outputs, so the team should describe earlier exposure accurately rather than claiming repository-wide test blindness. They were not modified or used to select these models.

## Glossary and references

| Term | Meaning |
|---|---|
| Affinity | Measured drug–target binding/activity score |
| Warm-start / cold-target | Known individual IDs / a target with no training affinities |
| Latent factor | Learned numerical coordinate, not an explicitly encoded biological property |
| F | Number of coordinates per drug/target vector; bounds interaction rank along with D and T |
| Bias | Learned drug/target offset from the fixed training mean |
| Dot product | Sum of products of corresponding vector entries |
| Frobenius squared norm | Sum of squared entries of a matrix |
| Residual | Prediction minus actual affinity |
| MSE / objective | Mean squared error / half training MSE plus normalized L2 penalty |
| Lambda | Coefficient of the global parameter squared-norm penalty |
| Gradient | Local derivative of objective with respect to a coefficient |
| Alpha | Adam's base learning rate |
| Adam moments | Moving averages of gradients and squared gradients |
| Batch / epoch | One update's observations / one pass through training observations |
| Patience | Consecutive non-improving validation epochs before stopping |
| CI | Global ordering agreement for observations with unequal actual scores |
| Hit@K | Fraction of eligible ligands whose predicted top-K contains a true-best target |
| Checkpoint | Saved inference parameters, mappings, settings and metadata |
| Provenance | Recorded source/data hashes and lineage linking evidence to its origin |

Primary repository evidence: [REPORT.md](results/biased_mf/REPORT.md), [PROTOCOL.md](results/biased_mf/PROTOCOL.md), [data preparation](data_preparation.ipynb) and the linked source/artifact files. The [project brief](../Project%202%20DAVIS%20and%20KIBA%20drug%E2%80%93target%20binding%281%29.pdf) supplies task requirements and original benchmark counts.

Background: [DeepDTA DAVIS data](https://github.com/hkmztrk/DeepDTA/tree/master/data/davis), [DeepDTA KIBA data](https://github.com/hkmztrk/DeepDTA/tree/master/data/kiba), and Kingma & Ba's [Adam paper](https://arxiv.org/abs/1412.6980). Benchmark background does not imply this implementation is a DeepDTA neural model. The inspected local CSVs do not identify which upstream download route was originally used.
