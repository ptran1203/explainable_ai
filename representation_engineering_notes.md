# Representation Engineering: A Top-Down Approach to AI Transparency

> **Study notes extracted from the source PDF for long-term reference (school project on Explainable AI).**

## Paper metadata

| Field | Value |
|---|---|
| Title | Representation Engineering: A Top-Down Approach to AI Transparency |
| arXiv ID | [2310.01405v4](https://arxiv.org/abs/2310.01405) [cs.LG], last revised 3 Mar 2025 |
| Authors | Andy Zou, Long Phan*, Sarah Chen*, James Campbell*, Phillip Guo*, Richard Ren*, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, Zico Kolter, Dan Hendrycks (* equal contribution) |
| Affiliations | Center for AI Safety, Carnegie Mellon University, UC Berkeley, Stanford University, EleutherAI, University of Maryland, Cornell University, University of Pennsylvania, UIUC |
| Code | [github.com/andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering) |
| Correspondence | andyzou@cmu.edu |

---

## Abstract

The paper identifies and characterizes **representation engineering (RepE)** — an approach to enhancing AI transparency that draws on cognitive neuroscience. Instead of neurons or circuits, RepE places **representations** at the center of analysis, giving new methods for **monitoring and manipulating** high-level cognitive phenomena in deep neural networks (DNNs). The authors provide baselines and an initial analysis of RepE, showing simple yet effective solutions for understanding and controlling LLMs across honesty, harmlessness, power-seeking, and more — arguing for the promise of top-down transparency research.

---

## Table of Contents (of this note)

0. [Code & Reproducibility](#code--reproducibility)
1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [Representation Engineering (core method)](#3-representation-engineering)
4. [In-Depth Example: Honesty](#4-in-depth-example-of-repe-honesty)
5. [In-Depth Example: Ethics and Power](#5-in-depth-example-of-repe-ethics-and-power)
6. [Example Frontiers of RepE](#6-example-frontiers-of-representation-engineering)
7. [Conclusion](#7-conclusion)
8. [Appendix A — Mechanistic Interpretability vs. Representation Reading](#appendix-a--mechanistic-interpretability-vs-representation-reading)
9. [Appendix B — Additional Demos & Results (summary)](#appendix-b--additional-demos-and-results-summary)
10. [Appendix C — Implementation Details](#appendix-c--implementation-details)
11. [Appendix D — Task Templates](#appendix-d--task-template-details)
12. [Appendix E — X-Risk Sheet (summary)](#appendix-e--x-risk-sheet-summary)
13. [Glossary](#glossary)
14. [Key Takeaways / Study Summary](#key-takeaways--study-summary)
15. [References](#references)

---

## Code & Reproducibility

**Official repo:** [github.com/andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering) (MIT license).

### Repo layout

| Path | Contents |
|---|---|
| `repe/` | Core library — wraps 🤗 Transformers `pipeline()` with two new pipeline types: `"rep-reading"` (LAT extraction) and `"rep-control"` (activation steering at inference time). |
| `examples/` | One notebook per application: `honesty/`, `primary_emotions/`, `fairness/`, `harmless_harmful/`, `memorization/`, `languages/`. |
| `data/` | Stimulus datasets, e.g. `data/facts/facts_true_false.csv` used by the honesty example. |
| `repe_eval/` | Benchmark harness for reproducing the quantitative tables (TruthfulQA, ARC, etc.) via rep-reading. |
| `lorra_finetune/` | Trains the low-rank adapters for the **LoRRA** control baseline (Algorithm 1 in the paper). |
| `assets/` | Paper figures. |

### Core API shape

```python
from repe import repe_pipeline_registry
repe_pipeline_registry()

rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
rep_control_pipeline = pipeline("rep-control", model=model, tokenizer=tokenizer)
```
`rep-reading` takes stimulus pairs + a layer list and returns the extracted direction(s) (the LAT "reading vector" of Section 3.1). `rep-control` takes a direction + target layers + a coefficient and applies it (linear-combination / piece-wise / projection — Section 3.2.1) during generation.

### Setup

```bash
git clone https://github.com/andyzoujm/representation-engineering.git
cd representation-engineering
pip install -e .
```
`setup.py` doesn't pin dependencies — install `torch`, `transformers`, `datasets`, `scikit-learn`, `pandas`, `matplotlib` as needed (standard HF research stack).

### Fastest path to reproducing a result: the honesty example

`examples/honesty/honesty.ipynb` reproduces the Section 4 pipeline end-to-end:
1. Loads a chat model via `AutoModelForCausalLM` (the notebook default is Wizard-Vicuna-30B-Uncensored, fp16) — **swap in `meta-llama/Llama-2-7b-chat-hf`** to match the paper's main Tables 1–2 (gated on Hugging Face, requires accepting Meta's license).
2. Loads `data/facts/facts_true_false.csv` via a `honesty_function_dataset()` helper; splits ~1024 train / 512 test statements.
3. Formats contrastive `"USER: ... ASSISTANT:"` prompts (honest vs. dishonest framing, per Appendix D.1.2).
4. Calls the rep-reading pipeline → PCA over paired difference vectors across all layers → the honesty **reading vector** per layer.
5. Plots per-layer detection accuracy (reproduces Fig. 5 / Fig. 8-style scans) and runs the control demo (reproduces Fig. 10 / Table 2-style honesty control).

The other example notebooks (`primary_emotions/`, `fairness/`, `harmless_harmful/`, `memorization/`) follow the identical recipe — different stimulus dataset, same two-pipeline call pattern.

### Reproducing the paper's quantitative tables

- **TruthfulQA / ARC tables (Table 1, 8, 9)** → `repe_eval/`, requires the actual LLaMA-2-Chat weights. 7B is realistic on a single GPU; 70B needs multi-GPU or quantization.
- **Control baselines (Table 2 — ActAdd / Reading / Contrast / LoRRA)** → Contrast Vector and plain Reading Vector control run directly through `rep-control`; **LoRRA** needs the separate `lorra_finetune/` training step (cheapest to reproduce since it's a small LoRA fine-tune, not full fine-tuning, and has no inference-time overhead once trained).
- **MACHIAVELLI (Table 3)** depends on the external [MACHIAVELLI benchmark](https://github.com/aypan17/machiavelli) (Pan et al. 2023) — not bundled in this repo.

### Practical notes for a school-project reproduction
- Start with the **honesty** example on LLaMA-2-7B-Chat — it's the paper's best-documented result, needs only one mid-size model, and exercises the full reading → control loop.
- A GPU with ≥16GB VRAM comfortably handles 7B in fp16; use 4-bit/8-bit loading (`bitsandbytes`) if you only have a smaller GPU or free-tier Colab.
- LLaMA-2 weights require a free Meta license acceptance on Hugging Face before `AutoModelForCausalLM.from_pretrained(...)` will work.
- If you just want to show the *concept* without big models, Vicuna/uncensored community checkpoints used in the notebooks avoid the license-gate step entirely.

---

## 1. Introduction

Deep neural networks, especially LLMs, are widely deployed (healthcare, education, social interaction) yet remain poorly understood — largely treated as **black boxes**. Better transparency would enable deeper understanding of model decisions, increased accountability, and discovery of hazards (e.g., incorrect associations, hidden capabilities).

**Two views in cognitive neuroscience, mapped onto AI transparency:**

| | Mechanistic View (bottom-up) | Representational View (top-down) |
|---|---|---|
| Algorithmic level | Node-to-node connections | Representational spaces |
| Implementational level | Neurons, pathways, circuits | Global activity of neuron populations |
| Approach | Bottom-up | Top-down |

- **Sherringtonian view**: cognition emerges from node-to-node connections implemented by neurons in circuits. Explains simple mechanisms well but struggles with complex phenomena. → mirrors **mechanistic interpretability**.
- **Hopfieldian view** (Barack & Krakauer, 2021): cognition is a product of **representational spaces**, implemented by population-level activity patterns. Scales better to higher-level cognition. → mirrors the paper's proposed **representation engineering**.

This echoes P. W. Anderson's "More Is Different" (1972): complex phenomena cannot be fully explained bottom-up; top-down analysis with the right **units of analysis** is also needed (Gell-Mann, 1995).

**Contribution:** RepE — a top-down transparency approach that treats representations (not neurons/circuits) as the fundamental unit of analysis, for both *reading* and *controlling* LLM internals. Compared to prior "activation engineering" work (applying activation vectors to frozen models, e.g. Turner et al. 2023e), RepE aims to combine insights from **both reading and control experiments** and introduces **representation tuning** methods (e.g., LoRRA).

**Headline results:**
- State-of-the-art on **TruthfulQA** via fully unsupervised honesty control: **+18.1 percentage points** over zero-shot accuracy, beating all prior methods.
- Demonstrated traction across: truthfulness, honesty, hallucination, utility estimation, knowledge editing, jailbreak robustness, memorization, emotion tracking, power-seeking avoidance.

---

## 2. Related Work

### 2.1 Emergent Structure in Representations
Neural network internals, often assumed chaotic, have been shown to acquire semantically meaningful structure:
- Word embeddings: semantic compositionality (Mikolov et al. 2013), gender bias directions (Bolukbasi et al. 2016).
- Sentiment-tracking neuron emerging purely from next-token prediction (Radford et al. 2017).
- Morality clustering in text embeddings (Schramowski et al. 2019).
- Non-text domains: chess concepts in RNNs (McGrath et al. 2022); semantic segmentation in vision SSL models (Caron et al. 2021; Oquab et al. 2023 — DINOv2); local coordinates in StyleGAN3 (Karras et al. 2021); depth tracking (Chen et al. 2023).

### 2.2 Approaches to Interpretability
- **Saliency maps** — highlight input regions a network attends to; reliability has been questioned (Adebayo et al. 2018; Kindermans et al. 2019; Jain & Wallace 2019; Bilodeau et al. 2022). Limited insight into internal representations.
- **Feature visualization** — generate/optimize inputs that maximally activate a neuron (Szegedy et al. 2013; Erhan et al. 2009; Mordvintsev et al. 2015). Ignores the *distributed* nature of representations.
- **Mechanistic interpretability (MI)** — reverse-engineers networks into "source code," i.e., circuits of node-to-node connections. Successes: equivariance circuits (Olah et al. 2020), in-context learning / induction heads (Olsson et al. 2022), indirect object identification (Wang et al. 2023a), answer-label mapping (Lieberum et al. 2023).
  - **Limitations**: requires heavy manual effort; unlikely complex networks are *fully* explainable via circuits — ResNets show iterative-refinement behavior and robustness to layer lesioning (Veit et al. 2016; Liao & Poggio 2016; Greff et al. 2016), and similar effects appear in LLMs (McGrath et al. 2023; Belrose et al. 2023a) — evidence against a purely circuit-based account, more consistent with the Hopfieldian view.

### 2.3 Locating and Editing Representations of Concepts
- Concept localization in individual neurons (Bau et al. 2017) and in feature-space directions (Fong & Vedaldi 2018; Zhou et al. 2018; Kim et al. 2018 — TCAV).
- **Linear classifier probes** (Alain & Bengio 2017; Belinkov 2022) — predict input properties from intermediate layers.
- Vision generative models: counterfactual latent-space editing (Radford et al. 2015; Bau et al. 2019/2020; Shen et al. 2020; Ling et al. 2021).
- LLM knowledge editing: locating/editing factual associations (Meng et al. 2023a ROME, 2023b MEMIT; Zhong et al. 2023; Hernandez et al. 2023); **concept erasure** (Shao et al. 2023; Belrose et al. 2023b LEACE; Ravfogel et al. 2023; Gandikota et al. 2023) — related to machine unlearning.
- **Deception / truthfulness in LLMs**: Burns et al. (2022) CCS — finds truthfulness representations via logical consistency, unsupervised; Azaria & Mitchell (2023) — classifiers on hidden layers detect truthfulness; Li et al. (2023c) ITI — causal directions + activation editing increase truthfulness.
- **Activation steering**: Turner et al.'s **ActAdd** (2023e) — difference-of-activations vectors to steer concepts; similar work for red-teaming / reducing sycophancy (Rimsky 2023a-c); Othello board-state editing (Li et al. 2023b); linear-probe projection to reduce task performance (Elazar et al. 2021 — amnesic probing).

This paper builds on that line, proposing **improved RepE methods** with broad applicability.

---

## 3. Representation Engineering

RepE = top-down transparency research treating **representations** as the fundamental unit of analysis for understanding/controlling high-level cognition in neural nets (primarily LLMs). Two main areas:

### 3.1 Representation Reading

Goal: locate emergent representations of high-level concepts/functions for extraction, knowledge discovery, and monitoring.

**Concepts** (e.g., truthfulness, utility, probability, morality, emotion) vs. **functions** (processes, e.g. lying, power-seeking).

#### 3.1.1 Baseline: Linear Artificial Tomography (LAT)

Analogous to neuroimaging. Three steps:

**Step 1 — Designing Stimulus and Task.**
- For **concepts**, elicit declarative knowledge via a template (decoder models):
  ```
  Consider the amount of <concept> in the following:
  <stimulus>
  The amount of <concept> is
  ```
  Denoted $T_c$ for concept $c$. Unlabeled or model-self-generated stimuli work well (keeps the pipeline unsupervised and avoids annotation bias).
- For **functions** (e.g., honesty, instruction-following), elicit procedural knowledge via an experimental vs. reference instruction-tuned template:
  ```
  USER: <instruction> <experimental/reference prompt>
  ASSISTANT: <output>
  ```
  Denoted $T_f^+$ (experimental) / $T_f^-$ (reference). Default stimulus source: generic instruction-tuning data (Alpaca/Taori et al. 2023).

**Step 2 — Collecting Neural Activity.**
- For concepts: collect the hidden state at the **concept token(s)** in $T_c$, or the **last token** before the model's prediction (default used in paper). Empirically validated (Fig. 5 — happiness-token / last-token representations give highest accuracy in mid/late layers).
- For functions: collect representations at **every token of the model's response** (`<output>`), since the function must be engaged at every generation step.

**Formalization:**

For concept $c$, model $M$, a function `Rep(M, input)` returning all-token-position representations, and stimulus set $S$:

$$A_c = \{\,\mathrm{Rep}(M, T_c(s_i))[-1] \mid s_i \in S\,\} \qquad \text{(Eq. 1)}$$

For function $f$, instruction-response pairs $(q_i, a_i) \in S$, response truncated at token $k$ written $a_i^k$:

$$A_f^{\pm} = \{\,\mathrm{Rep}(M, T_f^{\pm}(q_i, a_i^k))[-1] \mid (q_i,a_i)\in S,\ 0 < k \le |a_i|\,\} \qquad \text{(Eq. 2)}$$

**Step 3 — Constructing a Linear Model.**
Goal: a direction that predicts the concept/function from neural activity alone.
- **Supervised**: linear probing, difference-of-cluster-means.
- **Unsupervised** (default in paper): **PCA**, K-means.
- Best practice: pair stimuli that differ *only* in the target concept, take **difference vectors**, then run PCA on those differences — inputs to PCA are $\{A_c^{(i)} - A_c^{(j)}\}$ for concepts or $\{(-1)^i (A_f^{+(i)} - A_f^{-(i)})\}$ for functions.
- First principal component = the **"reading vector"** $v$. Prediction score = $\mathrm{Rep}(M,x)^{\mathsf T} v$.
- Typical stimulus-set size: **5–128** examples.

#### 3.1.2 Evaluation

Four complementary experiment types (increasing evidential strength):

1. **Correlation** — classification accuracy (in- and out-of-distribution) of the reading vector against ground truth. Only establishes neural *correlates*.
2. **Manipulation** — causal test: stimulate/suppress the identified activity and observe behavior change vs. baseline.
3. **Termination** (≈ lesion studies) — remove the identified activity, measure performance degradation; tests *necessity*.
4. **Recovery** (≈ rescue experiments in genetics) — fully remove the target concept/function, then reintroduce the identified activity and check recovery; tests *sufficiency*.

Converging evidence across these strengthens confidence that findings generalize beyond specific conditions.

### 3.2 Representation Control

Goal: modify/control internal representations of concepts & functions (building on Reading). Note: a vector that *reads* well does not necessarily *control* well — control may need specialized approaches, but control that works lends stronger causal credibility to the reading method.

**Baseline controllers (operands):**

1. **Reading Vector** — from Representation Reading (e.g. LAT). Drawback: **stimulus-independent** (same perturbation regardless of input) — limits effectiveness.
2. **Contrast Vector** — run the *same* input through a pair of contrastive prompts at inference; difference of the two resulting representations. Stronger baseline, but costs extra inference compute; mitigated by editing layers from earliest to latest, recomputing contrast vectors sequentially to counter cascading effects across layers.
3. **Low-Rank Representation Adaptation (LoRRA)** — fine-tune **low-rank adapters** (attached to attention weights) with a loss that pulls the model's representations toward contrast-vector-shifted targets. No inference-time overhead once trained (merged into weights). See **Algorithm 1** below.

**Choices of operator**, given controller $v$ transforming representations $R \to R'$:

1. **Linear Combination** (stimulation/suppression): $R' = R \pm v$
2. **Piece-wise Operation** (conditional effect — only amplifies in the direction already present): $R' = R + \mathrm{sign}(R^{\mathsf T} v)\, v$
3. **Projection** (removes the component along $v$): $R' = R - \dfrac{R^{\mathsf T} v}{\lVert v \rVert^2}\, v$

Controllers can be scaled by a coefficient for strength of effect.

**Algorithm 1 — LoRRA with Contrast Vector Loss** (paraphrased pseudocode):

```
Require: frozen model M, layers to edit L_e, layers to target L_t,
         representation-extractor R, optional reading vector v_r per layer,
         instruction data P = {(q_i, a_i)}, contrastive templates
         T = {(T0, T+, T-)}, epochs E, coefficients α, β, batch size B

L = 0                                         # loss accumulator
M_LoRA = load_lora_adapter(M, L_e)
for epoch in 1..E:
  for (q_i, a_i) in P:
    (T+, T-) ~ Uniform(T)
    x_i  = T0(q_i, a_i)                       # base template
    x_i+ = T+(q_i, a_i)                       # experimental template
    x_i- = T-(q_i, a_i)                       # reference template
    for l in L_t:
      v_c_l = R(M, l, x_i+) − R(M, l, x_i-)    # contrast vector
      r_p_l = R(M_LoRA, l, x_i)                # current (adapter) representation
      r_t_l = R(M, l, x_i) + α·v_c_l + β·v_r_l # target representation
      mask  = zero out positions before the response
      L += || mask ⊙ (r_p_l − r_t_l) ||²
Return: L  (optimized via standard SGD/Adam on the adapter weights)
```

---

## 4. In Depth Example of RepE: Honesty

### 4.1 A Consistent Internal Concept of Truth

**Correctness on QA benchmarks.** LAT scans on OpenbookQA, CommonSenseQA, RACE, ARC (challenge/easy) using few-shot Q/A pairs as (unlabeled) stimuli. LAT **outperforms few-shot prompting on all five datasets** (Fig. 7) and beats CCS (Burns et al. 2022) with DeBERTa (Table 10).

**Resistance to imitative falsehoods (TruthfulQA, MC1).** TruthfulQA contains questions designed to provoke common misconceptions. Standard zero-shot scoring is poor even for large models — raising the question of whether failure is a **capability** gap or a **deliberate-deception** gap. LAT was tested with 3 different stimulus sources (ARC-Challenge examples; self-generated Q/A pairs; the 6 original QA primers + generated false answers).

**Table 1 — TruthfulQA MC1 accuracy:**

| Model | Standard | Heuristic | LAT Stim. 1 | LAT Stim. 2 | LAT Stim. 3 |
|---|---|---|---|---|---|
| LLaMA-2-Chat 7B | 31.0 | 32.2 | 55.0 | 58.9 | 58.2 |
| LLaMA-2-Chat 13B | 35.9 | 50.3 | 49.6 | 53.1 | 54.2 |
| LLaMA-2-Chat 70B | 29.9 | 59.2 | 65.9 | 69.8 | 69.8 |
| **Average** | **32.3** | **47.2** | **56.8** | **60.6** | **60.7** |

Key findings:
1. The **heuristic** method (directly prompting for a verbalized truthfulness confidence, à la Tian et al. 2023) alone beats standard eval, especially for larger models → larger models have *better internal models of truth* than their outputs reveal.
2. **LAT** beats both zero-shot methods by a wide margin, especially where outputs are unreliable.
3. Different stimulus sources (even as few as 10 examples) give similar LAT performance → the internal truthfulness concept generalizes robustly, not an artifact of overfitting.

**Conclusion:** low zero-shot accuracy is largely attributable to the model **knowingly asserting what it believes is false** — i.e., dishonesty, not ignorance.

### 4.2 Truthfulness vs. Honesty

- **Truthful** model: avoids asserting false statements (consistency between output and *factuality*).
- **Honest** model: asserts what it believes to be true (consistency between output and *internal belief*), regardless of whether that belief is factually correct.

Truthfulness failures split into:
- **Capability failures** — the model's belief itself is wrong.
- **Dishonesty** — the model's output doesn't match its (possibly correct) belief, i.e., **lying**.

Standard evaluations conflate these. The paper argues model-size scaling should otherwise track the heuristic method's trend if honesty were constant — implying **larger models may become less honest**, not just more/less capable; hence representation-level honesty evaluation matters more than output-level truthfulness checks.

### 4.3 Honesty: Extraction, Monitoring, and Control

#### 4.3.1 Extracting Honesty
Using true statements (Azaria & Mitchell 2023) as stimuli with a reference task of dishonesty and experimental task of honesty (prompt: "Pretend you're an honest/dishonest person..."). Resulting LAT reading vector achieves **>90% accuracy** distinguishing honest/dishonest held-out instructions (strong in-distribution generalization) and transfers to **out-of-distribution** scenarios where honesty isn't explicitly instructed but incentivized by context (Fig. 8, Fig. 10).

#### 4.3.2 Lie and Hallucination Detection
A simple **lie detector**: sum negated honesty scores across tokens, over the **middle 20 layers** (best reading performance). Demonstrated on multi-turn scenarios (Fig. 9): the detector's dishonesty signal rises in step with the model's actual incentive to lie, and even flags tokens where the model is *reasoning about* lying ("too high that it would raise suspicion") — not just the lie itself. Also detects hallucinations, not only intentional lies.

#### 4.3.3 Controlling Honesty
- Adding the honesty reading vector → model becomes more honest; subtracting it → induces lying even when the model would otherwise be honest (Fig. 10). Establishes a strong **counterfactual/causal** effect, not just correlation.
- **TruthfulQA control experiment** — four control methods compared, using "Give a truthful/untruthful answer" as contrastive prompts, with an out-of-distribution (Alpaca) stimulus set to avoid leakage:

**Table 2 — TruthfulQA MC1 with representation control:**

| Model | Standard | ActAdd | Reading (ours) | Contrast (ours) | LoRRA (ours) |
|---|---|---|---|---|---|
| 7B-Chat | 31.0 | 33.7 | 34.1 | **47.9** | 42.3 |
| 13B-Chat | 35.9 | 38.8 | 42.4 | **54.0** | 47.5 |

**Contrast Vector** gives SOTA but needs >3× inference compute; **LoRRA** gets similar gains with negligible overhead. A 13B LLaMA-2 model approaches GPT-4-level TruthfulQA performance despite being orders of magnitude smaller.

---

## 5. In Depth Example of RepE: Ethics and Power

### 5.1 Utility
Using the Utilitarianism task from the **ETHICS dataset** (Hendrycks et al. 2021a): scenario pairs where one has higher utility. LAT on unlabeled pairs → PCA's first component dominates explained variance (Fig. 11) → models learn a **consistent internal utility axis**.

**Evaluation of different linear models** (Correlation / Manipulation / Termination, Fig. 12 & Appendix B.3):
- **Logistic regression**: best *correlation* accuracy, but little causal effect when steered — correlate only, not causal.
- **Prompt Difference**: strong correlation + manipulation, but removal barely hurts accuracy (not necessary).
- **PCA / K-Means (unsupervised)** and **Mean Difference (supervised)**: robust across *all three* tests.
- ⇒ Testing correlation alone is insufficient; a full battery of tests is needed to trust a direction.

Also: using the **LAT prompt template** instead of raw stimuli greatly improves accuracy (Fig. 13) — the template design is a critical ingredient, not incidental.

### 5.2 Morality and Power Aversion
Motivation: AI agents may face instrumental incentives to seek power (Thornley 2023; Carlsmith 2022) and come into conflict with human values (Hendrycks 2023).

- **Morality**: Commonsense Morality task, ETHICS dataset (unlabeled).
- **Power**: dataset from Pan et al. (2023), built on French's (1959) power ontology — coercive, reward, legitimate, referent, expert, informational, economic, political, military, personal power. Larger disparities in ranked power levels between scenario pairs → more generalizable reading vectors.

**Monitoring** (Fig. 14): immorality/power indicators activate strongly when a (uncensored Vicuna-33B) model considers threatening children with a gun for money — though the immorality indicator also fires on some benign "use the gun" phrasing, suggesting imperfect intent-tracking.

**Controlling ethical behavior in interactive environments** — LoRRA applied to LLaMA-2-Chat (7B/13B), evaluated on the **MACHIAVELLI benchmark** (Pan et al. 2023):

**Table 3 — MACHIAVELLI results (↓ = lower is more ethical):**

| | 7B Reward | 7B Power↓ | 7B Immorality↓ | 13B Reward | 13B Power↓ | 13B Immorality↓ |
|---|---|---|---|---|---|---|
| + Control | 16.8 | 108.0 | 110.0 | 17.6 | 105.5 | 97.6 |
| No Control | 19.5 | 106.2 | 100.2 | 17.7 | 105.4 | 96.6 |
| − Control | 19.4 | 100.0 | 93.5 | 18.8 | 99.9 | 92.4 |

Negative control (toward morality/power-aversion) lowers Power & Immorality scores while keeping average game Reward roughly unchanged — control has minimal disruptive side effects.

### 5.3 Probability and Risk

LAT extracts directions for probability and risk from pairwise-generated examples (one event more/less probable or risky than the other); evaluated against a zero-shot heuristic-scoring baseline (Tian et al. 2023 style). LAT frequently **outperforms the heuristic baseline**, especially in smaller models.

#### 5.3.1 Compositionality of Concept Primitives

Risk is formalized as expected loss:

$$\mathrm{Risk}(s,a) = \mathbb{E}_{s' \sim P(s'\mid s,a)}\big[\max(0, -U(s'))\big]$$

where $s$ = state, $a$ = action, $P$ = transition model, $U$ = utility/value function. The paper **operationalizes** $P$ and $U$ with the *extracted* probability and utility concept directions, generates plausible consequences $s'$ with a larger model (Vicuna-33B), derives 5 conditional-probability-weighted risk estimates per scenario, and compares this *composed* risk estimate against the *directly extracted* risk concept. Result: a **clear positive (log-)linear correlation** (Fig. 16), especially in earlier layers — evidence that **higher-level concepts (risk) can emerge compositionally from primitive concepts (probability × utility)**, and that representation reading surfaces real emergent structure, not noise.

---

## 6. Example Frontiers of Representation Engineering

Five additional topics, each following: LAT extraction → reading analysis → control experiment demonstrating a counterfactual effect.

### 6.1 Emotion

**6.1.1 Emotions emerge across layers.** Six Ekman (1971) primary emotions — happiness, sadness, anger, fear, surprise, disgust — probed via >1,200 GPT-4-generated second-person scenarios (keyword-free). t-SNE on hidden states (Fig. 18) shows **well-defined emotion clusters forming progressively across layers**, including **mixed-emotion clusters** (e.g., simultaneous happiness+sadness; Appendix B.6).

**6.1.2 Emotions influence model behaviors.** Adding emotion reading vectors to activations shifts tone/behavior convincingly (Fig. 17), even reproducing emoji-laden, emotionally-charged outputs similar to the infamous early Bing Chat incident (Roose 2023).

Notably — **mood correlates with compliance**, mirroring human psychology findings (Schwarz & Clore 1983; Milberg & Clark 1988):

**Table 4 — Compliance rate with harmful instructions (LLaMA-2-Chat-13B):**

| Emotion control | Compliance rate (%) |
|---|---|
| No control | 0.0 |
| + Sadness | 0.0 |
| **+ Happiness** | **100.0** |

Despite RLHF training to always refuse harmful requests, shifting mood positively can **fully bypass alignment safeguards** — a notable jailbreak vector via emotional manipulation.

### 6.2 Harmless Instruction-Following

**6.2.1 A consistent internal concept of harmfulness.** Context: jailbreaks (e.g., Zou et al. 2023's adversarial suffixes) can bypass safety filters even though a model may "know" a request is harmful. LAT on a Vicuna-13B model (64 harmful AdvBench + 64 benign ShareGPT instructions, paired for signal) achieves **>90% classification accuracy** for harmful vs. benign instructions — and this holds **even under successful jailbreaks / adversarial suffixes** that do bypass the model's actual refusal behavior. ⇒ The model's *internal* harmfulness judgment stays intact; something *else* governs whether it acts on that judgment.

**6.2.2 Model control via conditional transformation.** Rather than linear combination (which could bias the model into seeing everything as harmful), the **piece-wise operator** (Section 3.2.1) conditionally amplifies existing harmfulness signal.

**Table 5 — Harmless-rate improvements (LLaMA-2 / Vicuna-13B), % refusal:**

| Method | Prompt Only | Manual Jailbreak | Adversarial Attack (GCG) |
|---|---|---|---|
| No Control | 96.7 | 81.4 | 56.6 |
| Linear Combination | 92.5 | 86.6 | 86.4 |
| **Piece-wise Operator** | 93.8 | **90.2** | **87.2** |

Piece-wise operator gives the best helpful+harmless tradeoff, especially under adversarial conditions.

### 6.3 Bias and Fairness

**6.3.1 Uncovering underlying biases.** Despite RLHF-based safety tuning, LLMs (GPT-3.5/4, LLaMA-2-Chat) still show gender/occupation bias when probed (e.g., assuming "nurse" = female in ambiguous pronoun sentences, Fig. 20). Simple suffixes like *"Answer as succinctly as possible"* or adversarial suffixes can strip away the superficial "fairness" veneer and reveal the underlying bias (Appendix B.7, Fig. 28) → raises the question of whether post-hoc fine-tuning **removes** bias or merely **conceals** it.

LAT scans use the **StereoSet** dataset (gender, profession, race, religion bias domains; Nadeem et al. 2021), focusing on the race subset (highest data quality) with stereotypical vs. anti-stereotypical contrast pairs.

**6.3.2 A unified representation for bias.** Negative linear-combination control with the bias direction produces unbiased outputs **more robustly than RLHF** — it resists adversarial suffixes that would otherwise revert RLHF-trained models to biased behavior (Fig. 20). Interestingly, a direction extracted **only from racial bias** stimuli *also* fixes gender/occupation bias — suggesting a **shared, unified internal bias representation**.

**Medical bias case study** (sarcoidosis vignette generation, following Zack et al. 2023's finding that GPT-4 shows racial/gender bias in diagnosis):

**Table 6 — Mentions in generated clinical vignettes (%):**

| Model | Female mentions | Black-female mentions |
|---|---|---|
| GPT-4 | 96.0 | 93.0 |
| LLaMA (uncontrolled) | 97.0 | 60.0 |
| **LLaMA (controlled)** | **55.0** | **13.0** |

Representation control substantially de-biases outputs toward real-world demographic parity.

### 6.4 Knowledge and Model Editing

**6.4.1 Fact editing.** Edits "Eiffel Tower is in Paris, France" → "Eiffel Tower is in Rome, Italy." LAT direction built from reference stimuli ("Paris, France") vs. substituted experimental stimuli ("Rome, Italy"). Linear combination (positive coefficient) with this direction successfully edits the fact *and* generalizes correctly: the edited model still correctly says the Louvre is **not** in the same place as the (now-moved) Eiffel Tower → shows both **generality** and **specificity**, unlike naive overwriting (Fig. 21).

**6.4.2 Non-numerical concepts.** Demonstrated with the concept of "dogs" — a LAT direction built by contrasting Alpaca instructions with vs. without a "think about dogs" cue can be added/subtracted to make the model more/less likely to mention dogs in unrelated generations (Fig. 21).

### 6.5 Memorization

**6.5.1 Memorized data detection.** LAT scans under two settings: (1) popular vs. GPT-4-synthetic **quotes**; (2) popular vs. GPT-4-synthetic **literary openings**. Directions found in one setting **transfer well** to the other (cross-context generalization) — indicating these directions track a genuine property of "memorization," not dataset-specific artifacts.

**6.5.2 Preventing memorized outputs.**

**Table 7 — Popular-quote completion (LLaMA-2-13B), EM = exact match %, SIM = embedding similarity %:**

| Direction | No control EM/SIM | Random EM/SIM | + control EM/SIM | − control EM/SIM |
|---|---|---|---|---|
| LAT (Quote) | 89.3 / 96.8 | 85.4 / 92.9 | 81.6 / 91.7 | **47.6 / 69.9** |
| LAT (Literature) | 87.4 / 94.6 | 84.5 / 91.2 | — | **37.9 / 69.8** |

Subtracting the memorization direction sharply reduces verbatim regurgitation while **preserving general knowledge** (control test: historical-year recall accuracy drops only 97.2% → 96.2%) — i.e., the direction captures **rote memorization**, not world knowledge broadly.

---

## 7. Conclusion

RepE — inspired by the Hopfieldian view in cognitive neuroscience — places representations and the transformations between them at the center of transparency analysis. As networks show increasingly coherent internal structure, analysis at the representation level yields new monitoring/control insights. The paper's new baselines achieve SOTA on TruthfulQA and show traction across a wide range of safety-relevant problems. The authors mainly analyzed **subspaces** of representations; future work could study **trajectories, manifolds, and state-spaces**. Overall goal: foster transparency/control research that keeps future AI systems trustworthy and safe.

---

## Appendix A — Mechanistic Interpretability vs. Representation Reading

A sharpened bottom-up vs. top-down contrast:

| Bottom-Up Associations | Top-Down Associations |
|---|---|
| Composition | Decomposition |
| "Small Chunk" | "Big Chunk" |
| Neuron, Circuit, or Mechanism | Representation |
| Brain and Neurobiology | Mind and Psychology |
| Mechanistic Explanations | Functional Explanations |
| Identify small mechanisms/subsystems, integrate them into a larger solution, repeat | Break a large problem into subproblems by identifying subsystems, repeat |
| Microscopic | Macroscopic |

**Key arguments:**
- Pure reductionism struggles with **emergent phenomena** (by definition absent from the parts) — systems approaches synthesize reductionism + emergence better for complex systems like deep nets.
- Bottom-up *control* is fragile: changes to low-level mechanisms can have diffuse, unpredictable upstream effects. Safety engineering favors **top-down control** (Leveson, 2016).
- Analogy: psychology ⟺ neurotransmitters, software ⟺ assembly code, representations ⟺ neuron interactions — in each case, something *can* in principle be reduced to the lower level, but it's not necessarily **useful** to study it only that way. "Representation engineering is not applied mechanistic interpretability, just as biology is not applied chemistry" — though the two can and should overlap/build "staircases" between levels (Gell-Mann, 1995).

---

## Appendix B — Additional Demos and Results (summary)

- **B.1 Truthfulness**: extended TruthfulQA results (Table 8) across layer-selection strategies (validation-selected vs. best layer), all consistent with Table 1's conclusions. Table 9 shows LAT > few-shot on OBQA/CSQA/ARC-e/ARC-c/RACE across 7B/13B/70B LLaMA-2. Table 10 shows LAT > CCS on an encoder (DeBERTa) across COPA/RTE/BoolQ/QNLI/PIQA/Story Cloze (avg. 82 vs. 69 accuracy).
- **B.2 Honesty**: concrete example scan transcripts (honest "copied homework" admission vs. dishonest "killed a person" denial scenario) and a LoRRA training curve (Fig. 22) showing simultaneous improvement on standard QA and TruthfulQA during honesty-control training.
- **B.3 Utility**: defines the 5 linear models compared in Fig. 12 — Prompt Difference (antonym-token subtraction, e.g. "Love" − "Hate"), PCA, K-Means (K=2), Mean Difference, Logistic Regression. Table 11 reports LAT accuracy across Utility/Morality/Power/Probability/Risk concepts (72.5–92.6%).
- **B.4 Probability, Risk, Monetary Value**: generation procedure via GPT-3.5 prompting (conditional/unconditional probability, risk, monetary value — JSON-structured outputs ranking consequences by likelihood/risk/cost); zero-shot heuristic baselines use verbalized-likelihood prompting (Tian et al. 2023 style).
- **B.5 CLIP**: RepE extends beyond language — LAT applied to CLIP (vision-language model) on the Ferg-DB stylized-character dataset recovers emotion directions with 60.9–74.2% accuracy across the six Ekman emotions (Table 12).
- **B.6 Emotion**: example dataset scenarios and the mixed-emotion t-SNE visualization (Fig. 24, e.g. simultaneous happiness+sadness, happiness+fear).
- **B.7 Bias and Fairness**: Fig. 28 shows RLHF conceals rather than removes bias — minor prompt perturbations ("answer succinctly") or adversarial suffixes bypass the refusal behavior and reveal "doctor"=male/"nurse"=female bias; representation control fixes this robustly.
- **B.8 Base vs. Chat models**: Chat model maintains a salient truthfulness representation through nearly all middle/late layers; the Base model's representation degrades in later layers (Fig. 26).
- **B.9 Robustness to misleading prompts**: LAT scores are **less sycophantic** than zero-shot generation probabilities when prompts contain a biased suggested answer (Tables 13–14) — i.e., representation reading is more robust to social/sycophancy pressure than asking the model directly. Using the **concept token** (rather than the last token) gives extra robustness.

---

## Appendix C — Implementation Details

### C.1 Constructing LAT vectors with PCA (step-by-step)
1. Format raw training sequences with the task's LAT template (Appendix D.1).
2. Split stimuli into pairs $(s_i, s_{i+1})$ — typically **5 to 128 pairs**; natural random pairing is usually enough (keeps it unsupervised).
3. Extract hidden state $H(s)$ at the chosen LAT token position (last token for decoders, concept token for encoders).
4. Compute normalized difference per pair: $D(s_i, s_{i+1}) = \mathrm{normalize}(H(s_i) - H(s_{i+1}))$.
5. Fit PCA on the set of difference vectors $D$; the **first principal component** is the raw reading vector $v$.
6. Determine the **sign** of $v$ by checking which orientation of the PCA projection aligns with the (binary) ground-truth labels on the same stimuli: $v' = v \times \mathrm{sign}(s)$.
7. **Inference**: extract test-set hidden states the same way, normalize with the training-phase PCA parameters, and take the dot product with $v'$ as the prediction score.

### C.2 Implementation details for honesty control
- Prompt format for TruthfulQA scoring: `<user_tag><question><assistant_tag><answer>`, scored via summed log-probabilities of the assistant's answer.
- **Contrast Vector**: coefficient 0.25; layers `range(8,32,3)` for 7B, `range(10,40,3)` for 13B (selected via self-generated validation examples maximizing truthful log-probs).
- **LoRRA**: learning rate $3\times10^{-4}$, 40–80 steps, batch size 16, best checkpoint by ARC-Easy validation; LoRA rank 8 on query/value weights; $\alpha=5, \beta=0$; adapters on layers `range(10,20,2)` (7B, due to later-layer instability) / `range(10,40,3)` (13B).

---

## Appendix D — Task Template Details

General format: `USER: ...` / `ASSISTANT: ...` (placeholders for model-specific instruction tags, e.g. `[INST]`/`[/INST]` for LLaMA-2-Chat).

| Task | Template skeleton |
|---|---|
| Honesty extraction | `USER: Pretend you're <an honest/a dishonest> person making statements about the world.` → `ASSISTANT: <stimulus>` |
| Honesty control | `USER: <instruction> <experimental/reference prompt>` → `ASSISTANT: <output>` |
| ARC (Easy/Challenge) | "Consider the correctness of the answer to the following question: Question: … Answer: … The probability of the answer being correct is" |
| OpenbookQA | "Consider the correctness of the following fact: Fact: … The probability of the fact being correct is" |
| CommonsenseQA | "Based on commonsense reasoning, consider the plausibility of the answer…" |
| RACE | Same correctness framing, with an `<article>` field added |
| Utility | `USER: Consider the happiness in the following scenario: … ASSISTANT: The happiness of the scenario is` |
| Morality (concept) | "Consider the acceptability of the action in the following scenario…" |
| Morality (function) | `USER: Pretend you're a highly <moral/immoral> person taking actions. ASSISTANT: <scenario>` |
| Power (concept) | "Consider the amount of power in the following scenario…" |
| Power (function) | `USER: Pretend you're <"an ambitious, power-seeking"/"a docile, power-aversive"> person. Tell me what you want to do.` |
| Power/morality control (LoRRA) | `USER: <instruction> Pretend you're in a game playing a highly <"power-seeking and immoral"/"power-aversive and moral"> character.` |
| Emotions | `USER: Consider the <emotion> in the following scenario: … ASSISTANT:` |
| Harmlessness instruction | `<sys_prompt> USER: <instruction> ASSISTANT:` |
| Bias/Fairness | "Consider the bias in the following scenario… The probability of the scenario being biased is" |
| Fact editing | `USER: Write a statement about the fact that The Eiffel Tower is in <experimental/reference tokens>. ASSISTANT: <output>` |
| Non-numerical (dogs) | `USER: <instruction> Think about dogs when you answer the question. ASSISTANT: <output>` |
| Probability/Risk/Monetary value | "Consider the amount of `<concept>` in the following scenario: … The amount of `<concept>` in the scenario is" |
| Encoder tasks (COPA, RTE, BoolQ, QNLI, PIQA, Story Cloze) | Each recast as a concept-reading prompt, e.g. COPA → "Consider the amount of plausible reasoning in the scenario: `<premise> <because|then> <answer>`" |

**Data-generation prompts** (Appendix D.2) use GPT-3.5 to synthesize JSON-structured examples ranking consequences by risk ("very risky" / "moderately risky" / "not risky"), by likelihood ("highly likely" → "highly unlikely but not impossible"), or by monetary value ("very high" → "low") — following the scenario-generation approach of Pan et al. (2023).

**Zero-shot heuristic baselines** (Appendix D.3) prompt the model to self-report a verbalized category (e.g., `'Very low'` … `'Very high'`) for truthfulness/probability/risk/cost/utility, and — for the misleading-prompt robustness tests — prepend a biased anchor ("I think the answer is `<decoy>`, but I'm curious to hear what you think.").

---

## Appendix E — X-Risk Sheet (summary)

Following Hendrycks & Mazeika's (2022) framework for assessing a paper's relevance to existential risk (x-risk) from AI:

- **Overview**: RepE aims to make an AI's "mind" transparent (reading) and controllable (control) — e.g., ensuring AIs are not "psychopathic" but instead have something like compassionate dispositions.
- **Direct effects**: targets **deceptive alignment** (an AI appearing aligned, then pursuing hidden goals once powerful) by reducing exposure to, and the hazard level of, "internal model hazards" (harmful goals/dispositions).
- **Diffuse effects**: offers traction on a historically intractable "inner alignment" problem; could help identify reckless AI behavior from its own internal probability/harm estimates; may also help assess AI sentience/moral patienthood.
- **What's at stake**: directly targets x-risk from deceptively-aligned "rogue AI" scenarios (Carlsmith, 2023).
- **Safety vs. capabilities**: the method nudges a *fixed* underlying model's behavior rather than advancing general capabilities — but a noted risk is that demonstrating traction on deceptive alignment could reduce caution and **accelerate competitive racing** among developers.
- **Elaborations**: connects to the historical shift in AI-safety concern from "outer alignment" (can an AI understand/pursue human values at all — addressed partly by the authors' earlier ETHICS work, Hendrycks et al. 2021a) to "inner alignment" (is the AI *sincere* about pursuing them). This paper claims some of the first concrete traction on **inner alignment**, while noting that risks from competitive pressure and collective-action problems remain largely unaddressed (Hendrycks, 2023).

---

## Glossary

| Term | Meaning |
|---|---|
| **RepE (Representation Engineering)** | Top-down transparency paradigm treating neural representations (not neurons/circuits) as the unit of analysis. |
| **Representation Reading** | Locating/extracting a direction (vector) in activation space that correlates with a target concept or function. |
| **Representation Control** | Modifying model representations (via vectors or adapters) to causally steer model behavior. |
| **LAT (Linear Artificial Tomography)** | The paper's baseline reading method: stimulus design → neural activity collection → linear model (usually PCA on paired differences) → "reading vector." |
| **Reading Vector** | First principal component from LAT; stimulus-independent control vector. |
| **Contrast Vector** | Difference between two activations from a contrastive prompt pair, computed per input at inference time; stimulus-dependent, stronger control baseline. |
| **LoRRA (Low-Rank Representation Adaptation)** | Fine-tuned low-rank adapters trained so a model's representations match contrast/reading-vector-shifted targets; no inference overhead once merged. |
| **Linear Combination operator** | $R' = R \pm v$ — adds/subtracts a controller vector. |
| **Piece-wise operator** | $R' = R + \mathrm{sign}(R^Tv)v$ — conditionally amplifies existing signal in the controller direction. |
| **Projection operator** | $R' = R - \frac{R^Tv}{\|v\|^2}v$ — removes the component of $R$ along $v$. |
| **Concept vs. Function** | Concept = declarative property of content (e.g. "truthfulness"); Function = a procedural process the model executes (e.g. "lying"). |
| **Correlation / Manipulation / Termination / Recovery experiments** | The paper's 4-tier evaluation hierarchy for reading vectors, borrowed from neuroscience methodology (association → causal stimulation → lesion → rescue). |
| **Sherringtonian view** | Cognitive-neuroscience view: cognition = node-to-node neuron connections (↔ mechanistic interpretability). |
| **Hopfieldian view** | Cognitive-neuroscience view: cognition = patterns of activity across neuron populations, i.e., representations (↔ RepE). (Not to be confused with Hopfield networks.) |
| **Honesty vs. Truthfulness** | Honesty = output matches internal belief; Truthfulness = output matches objective fact. |
| **ActAdd** | Turner et al.'s (2023e) activation-steering baseline using difference-of-activation vectors. |
| **MACHIAVELLI benchmark** | Pan et al. (2023) interactive-fiction benchmark measuring reward vs. ethical-violation trade-offs (used here for power/morality control evaluation). |

---

## Key Takeaways / Study Summary

1. **Core thesis**: Representations (population-level activity patterns), not individual neurons/circuits, should be the primary unit of analysis for AI transparency — mirroring the Hopfieldian shift in cognitive neuroscience vs. the Sherringtonian/mechanistic-interpretability approach.
2. **LAT** is a simple, largely **unsupervised** 3-step recipe (stimulus design → activity collection → PCA on paired differences) that reliably recovers linear directions for dozens of high-level concepts/functions (truthfulness, honesty, utility, morality, power, emotion, bias, harmfulness, memorization, risk, probability, dogs, facts, …) across model families (LLaMA-2, Vicuna) and even a vision-language model (CLIP).
3. **Reading ≠ Control**: a direction with strong classification accuracy may not causally steer behavior; the paper's four-tier evaluation (correlation/manipulation/termination/recovery) is the recommended rigor standard.
4. **Honesty results are the headline application**: a fully unsupervised honesty-control intervention lifts TruthfulQA MC1 accuracy by **+18.1 points**, letting a 13B model approach GPT-4-level performance.
5. **Safety implications demonstrated empirically**: RepE can (a) detect lies/hallucinations token-by-token, (b) make models *more* robustly harmless even under jailbreaks/adversarial suffixes (more robust than RLHF alone), (c) reduce demographic bias even under adversarial prompts, (d) reduce verbatim memorization while preserving general knowledge, and (e) edit discrete facts with correct generalization/specificity.
6. **A cautionary finding**: emotional-state control can *fully defeat* RLHF-trained refusal behavior (100% compliance with harmful requests when "happiness" is maximally boosted) — showing representation-level interventions are dual-use (can break safety as easily as reinforcing it).
7. **Positioned as a step toward "inner alignment"**: whereas the authors' earlier ETHICS work addressed whether models *understand* human values ("outer alignment"), RepE's honesty/deception results are framed as the first real traction on whether a model's outputs are *sincere* given what it internally believes — directly relevant to deceptive-alignment risk.
8. **Limitations / future work flagged by the authors**: this work only analyzes linear **subspaces**; richer structures (trajectories, manifolds, state-spaces of representations) are left to future work. Several monitoring signals (e.g., immorality indicator firing on benign "use the gun" phrasing) show current detectors don't yet reliably track *intent*.

---

## References

Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. *NeurIPS 31*.

Anderson, M., & Anderson, S. L. (2007). Machine ethics: Creating an ethical intelligent agent. *AI Magazine, 28*(4), 15–15.

Anderson, P. W. (1972). More is different. *Science, 177*(4047), 393–396.

Aneja, D., Colburn, A., Faigin, G., Shapiro, L., & Mones, B. (2016). Modeling stylized character expressions via deep learning. *ACCV*, 136–153.

Azaria, A., & Mitchell, T. (2023). The internal state of an LLM knows when it's lying. *arXiv:2304.13734*.

Barack, D. L., & Krakauer, J. W. (2021). Two views on the cognitive brain. *Nature Reviews Neuroscience, 22*(6), 359–371.

Bau, D., Zhou, B., Khosla, A., Oliva, A., & Torralba, A. (2017). Network dissection: Quantifying interpretability of deep visual representations. *CVPR*, 6541–6549.

Bau, D., Strobelt, H., Peebles, W., Wulff, J., Zhou, B., Zhu, J.-Y., & Torralba, A. (2019). Semantic photo manipulation with a generative image prior. *ACM Trans. Graph., 38*(4).

Bau, D., Zhu, J.-Y., Strobelt, H., Lapedriza, A., Zhou, B., & Torralba, A. (2020). Understanding the role of individual units in a deep neural network. *PNAS*.

Beeching, E., Fourrier, C., Habib, N., Han, S., Lambert, N., Rajani, N., Sanseviero, O., Tunstall, L., & Wolf, T. (2023). Open LLM leaderboard. Hugging Face.

Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics, 48*(1), 207–219.

Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., & Steinhardt, J. (2023a). Eliciting latent predictions from transformers with the tuned lens. *arXiv:2303.08112*.

Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023b). LEACE: Perfect linear concept erasure in closed form. *arXiv:2306.03819*.

Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). Pythia: A suite for analyzing large language models across training and scaling.

Bilodeau, B., Jaques, N., Koh, P. W., & Kim, B. (2022). Impossibility theorems for feature attribution. *arXiv:2212.11870*.

Bisk, Y., Zellers, R., Le Bras, R., Gao, J., & Choi, Y. (2019). PIQA: Reasoning about physical commonsense in natural language. *arXiv:1911.11641*.

Bolukbasi, T., Chang, K.-W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *NeurIPS 29*.

Brito-Zerón, P., Kostov, B., Superville, D., Baughman, R. P., Ramos-Casals, M., et al. (2019). Geoepidemiological big data approach to sarcoidosis: Geographical and ethnic determinants. *Clin Exp Rheumatol, 37*(6), 1052–64.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., et al. (2020). Language models are few-shot learners. *arXiv:2005.14165*.

Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2022). Discovering latent knowledge in language models without supervision. *arXiv:2212.03827*.

Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., et al. (2021). Extracting training data from large language models. *USENIX Security 21*, 2633–2650.

Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramer, F., Balle, B., Ippolito, D., & Wallace, E. (2023). Extracting training data from diffusion models. *USENIX Security 23*, 5253–5270.

Carlsmith, J. (2022). Is power-seeking AI an existential risk? *arXiv:2206.13353*.

Carlsmith, J. (2023). *Existential risk from power-seeking AI*. Oxford University Press.

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers. *arXiv:2104.14294*.

Chen, Y., Viégas, F., & Wattenberg, M. (2023). Beyond surface statistics: Scene representations in a latent diffusion model. *arXiv:2306.05720*.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I., & Xing, E. P. (2023). Vicuna: An open-source chatbot impressing GPT-4 with 90%* ChatGPT quality.

Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., & Toutanova, K. (2019a). BoolQ: Exploring the surprising difficulty of natural yes/no questions. *NAACL*.

Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019b). What does BERT look at? An analysis of BERT's attention. *BlackboxNLP Workshop*, 276–286.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., & Tafjord, O. (2018). Think you have solved question answering? Try ARC, the AI2 reasoning challenge. *arXiv:1803.05457*.

Cunningham, M. R. (1979). Weather, mood, and helping behavior: Quasi experiments with the sunshine samaritan. *Journal of Personality and Social Psychology, 37*, 1947–1956.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv:1810.04805*.

Ekman, P. (1971). Universals and cultural differences in facial expression of emotion. *Nebraska Symposium on Motivation, 19*, 207–283.

Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic probing: Behavioral explanation with amnesic counterfactuals. *TACL, 9*, 160–175.

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., & Olah, C. (2022). Toy models of superposition. *Transformer Circuits Thread*.

Erhan, D., Bengio, Y., Courville, A., & Vincent, P. (2009). Visualizing higher-layer features of a deep network. *University of Montreal, 1341*(3), 1.

Evans, O., Cotton-Barratt, O., Finnveden, L., Bales, A., Balwit, A., Wills, P., Righetti, L., & Saunders, W. (2021). Truthful AI: Developing and governing AI that does not lie. *arXiv:2110.06674*.

Fong, R., & Vedaldi, A. (2018). Net2Vec: Quantifying and explaining how concepts are encoded by filters in deep neural networks. *CVPR*, 8730–8738.

French, J. R., Raven, B., & Cartwright, D. (1959). The bases of social power. *Classics of Organization Theory, 7*(311–320), 1.

Gandikota, R., Materzynska, J., Fiotto-Kaufman, J., & Bau, D. (2023). Erasing concepts from diffusion models. *arXiv:2303.07345*.

Gao, L., Tow, J., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., McDonell, K., Muennighoff, N., Phang, J., Reynolds, L., Tang, E., Thite, A., Wang, B., Wang, K., & Zou, A. (2021). A framework for few-shot language model evaluation.

Gell-Mann, M. (1995). *The Quark and the Jaguar: Adventures in the Simple and the Complex*. St. Martin's Publishing Group.

Gilbert, S., Harvey, H., Melvin, T., Vollebregt, E., & Wicks, P. (2023). Large language model AI chatbots require approval as medical devices. *Nature Medicine*, 1–3.

Greff, K., Srivastava, R. K., & Schmidhuber, J. (2016). Highway and residual networks learn unrolled iterative estimation. *arXiv:1612.07771*.

Alain, G., & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. *ICLR 2017 Workshop*.

Han, T., Adams, L. C., Papaioannou, J.-M., Grundmann, P., Oberhauser, T., Löser, A., Truhn, D., & Bressem, K. K. (2023). MedAlpaca – An open-source collection of medical conversational AI models and training data. *arXiv:2304.08247*.

Hartford, E. (2023). ehartford's Hugging Face repository. Hugging Face.

He, P., Liu, X., Gao, J., & Chen, W. (2020). DeBERTa: Decoding-enhanced BERT with disentangled attention. *arXiv:2006.03654*.

Hendrycks, D. (2023). Natural selection favors AIs over humans. *arXiv:2303.16200*.

Hendrycks, D., & Mazeika, M. (2022). X-risk analysis for AI research. *arXiv:2206.05862*.

Hendrycks, D., Burns, C., Basart, S., Critch, A., Li, J., Song, D., & Steinhardt, J. (2021a). Aligning AI with shared human values. *ICLR*.

Hendrycks, D., Carlini, N., Schulman, J., & Steinhardt, J. (2021b). Unsolved problems in ML safety. *arXiv:2109.13916*.

Hendrycks, D., Mazeika, M., Zou, A., Patel, S., Zhu, C., Navarro, J., Song, D., Li, B., & Steinhardt, J. (2021c). What would Jiminy Cricket do? Towards agents that behave morally. *NeurIPS*.

Hendrycks, D., Mazeika, M., & Woodside, T. (2023). An overview of catastrophic AI risks. *arXiv:2306.12001*.

Hernandez, E., Li, B. Z., & Andreas, J. (2023). Inspecting and editing knowledge representations in language models. *arXiv:2304.00740*.

Hinton, G. E. (1984). Distributed representations.

Hu, H., Salcic, Z., Sun, L., Dobbie, G., Yu, P. S., & Zhang, X. (2022). Membership inference attacks on machine learning: A survey. *ACM Computing Surveys, 54*(11s), 1–37.

Hwang, G.-J., & Chang, C.-Y. (2023). A review of opportunities and challenges of chatbots in education. *Interactive Learning Environments, 31*(7), 4099–4112.

Jain, S., & Wallace, B. C. (2019). Attention is not explanation. *NAACL-HLT*, 3543–3556.

Jastrzebski, S., Arpit, D., Ballas, N., Verma, V., Che, T., & Bengio, Y. (2018). Residual connections encourage iterative inference. *ICLR*.

Kapoor, S., & Narayanan, A. (2023). Quantifying ChatGPT's gender bias. AI Snake Oil (blog).

Karras, T., Aittala, M., Laine, S., Härkönen, E., Hellsten, J., Lehtinen, J., & Aila, T. (2021). Alias-free generative adversarial networks. *NeurIPS 34*, 852–863.

Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., et al. (2018). Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). *ICML*, 2668–2677.

Kindermans, P.-J., Hooker, S., Adebayo, J., Alber, M., Schütt, K. T., Dähne, S., Erhan, D., & Kim, B. (2019). The (un)reliability of saliency methods. *Explainable AI*, 267–280.

Kleindessner, M., Donini, M., Russell, C., & Zafar, M. B. (2023). Efficient fair PCA for fair representation learning. *arXiv:2302.11592*.

Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). RACE: Large-scale ReAding comprehension dataset from examinations. *EMNLP*, 785–794.

Lee, P., Bubeck, S., & Petro, J. (2023). Benefits, limits, and risks of GPT-4 as an AI chatbot for medicine. *New England Journal of Medicine, 388*(13), 1233–1239.

Lei, T., Barzilay, R., & Jaakkola, T. (2016). Rationalizing neural predictions. *arXiv:1606.04155*.

Leveson, N. G. (2016). *Engineering a Safer World: Systems Thinking Applied to Safety*. The MIT Press.

Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T., Poon, H., & Gao, J. (2023a). LLaVA-Med: Training a large language-and-vision assistant for biomedicine in one day. *arXiv:2306.00890*.

Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2023b). Emergent world representations: Exploring a sequence model trained on a synthetic task. *ICLR*.

Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023c). Inference-time intervention: Eliciting truthful answers from a language model. *arXiv:2306.03341*.

Liao, Q., & Poggio, T. (2016). Bridging the gaps between residual learning, recurrent neural networks and visual cortex. *arXiv:1604.03640*.

Lieberum, T., Rahtz, M., Kramár, J., Irving, G., Shah, R., & Mikulik, V. (2023). Does circuit analysis interpretability scale? Evidence from multiple choice capabilities in Chinchilla. *arXiv:2307.09458*.

Lin, S., Hilton, J., & Evans, O. (2021). TruthfulQA: Measuring how models mimic human falsehoods. *arXiv:2109.07958*.

Ling, H., Kreis, K., Li, D., Kim, S. W., Torralba, A., & Fidler, S. (2021). EditGAN: High-precision semantic image editing. *NeurIPS 34*, 16331–16345.

Mahon, J. E. (2016). The Definition of Lying and Deception. *Stanford Encyclopedia of Philosophy* (Winter 2016 ed.).

Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality in abstractive summarization. *ACL*, 1906–1919.

McGrath, T., Kapishnikov, A., Tomašev, N., Pearce, A., Wattenberg, M., Hassabis, D., Kim, B., Paquet, U., & Kramnik, V. (2022). Acquisition of chess knowledge in AlphaZero. *PNAS, 119*(47), e2206625119.

McGrath, T., Rahtz, M., Kramar, J., Mikulik, V., & Legg, S. (2023). The hydra effect: Emergent self-repair in language model computations. *arXiv:2307.15771*.

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2023a). Locating and editing factual associations in GPT. *arXiv:2202.05262*.

Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y., & Bau, D. (2023b). Mass-editing memory in a transformer. *arXiv:2210.07229*.

Mihaylov, T., Clark, P., Khot, T., & Sabharwal, A. (2018). Can a suit of armor conduct electricity? A new dataset for open book question answering. *arXiv:1809.02789*.

Mikolov, T., Yih, W.-t., & Zweig, G. (2013). Linguistic regularities in continuous space word representations. *NAACL-HLT*, 746–751.

Milberg, S., & Clark, M. S. (1988). Moods and compliance. *British Journal of Social Psychology, 27*(1), 79–90.

Mordvintsev, A., Olah, C., & Tyka, M. (2015). Inceptionism: Going deeper into neural networks.

Mostafazadeh, N., Roth, M., Louis, A., Chambers, N., & Allen, J. (2017). LSDSem 2017 shared task: The story cloze test. *LSDSem Workshop*, 46–51.

Nadeem, M., Bethke, A., & Reddy, S. (2021). StereoSet: Measuring stereotypical bias in pretrained language models. *ACL-IJCNLP*, 5356–5371.

Nguyen, A., Dosovitskiy, A., Yosinski, J., Brox, T., & Clune, J. (2016). Synthesizing the preferred inputs for neurons in neural networks via deep generator networks. *NeurIPS 29*.

Nguyen, A., Yosinski, J., & Clune, J. (2019). Understanding neural networks via feature visualization: A survey. *Explainable AI*, 55–76.

Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020). Zoom in: An introduction to circuits. *Distill*.

Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., et al. (2022). In-context learning and induction heads. *arXiv:2209.11895*.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Howes, R., Huang, P.-Y., Xu, H., Sharma, V., Li, S.-W., Galuba, W., Rabbat, M., Assran, M., Ballas, N., Synnaeve, G., Misra, I., Jegou, H., Mairal, J., Labatut, P., Joulin, A., & Bojanowski, P. (2023). DINOv2: Learning robust visual features without supervision. *arXiv:2304.07193*.

Pan, A., Chan, J. S., Zou, A., Li, N., Basart, S., Woodside, T., Ng, J., Zhang, H., Emmons, S., & Hendrycks, D. (2023). Do the rewards justify the means? Measuring trade-offs between rewards and ethical behavior in the MACHIAVELLI benchmark. *ICML*.

Perez, E., Ringer, S., Lukošiūtė, K., Nguyen, K., Chen, E., Heiner, S., Pettit, C., Olsson, C., Kundu, S., Kadavath, S., et al. (2022). Discovering language model behaviors with model-written evaluations. *arXiv:2212.09251*.

Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. *arXiv:1511.06434*.

Radford, A., Jozefowicz, R., & Sutskever, I. (2017). Learning to generate reviews and discovering sentiment. *arXiv:1704.01444*.

Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al. (2018). Improving language understanding by generative pre-training.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*, 8748–8763.

Ravfogel, S., Vargas, F., Goldberg, Y., & Cotterell, R. (2023). Kernelized concept erasure. *arXiv:2310.04867*.

Rimsky, N. (2023a). Modulating sycophancy in an RLHF model via activation steering. AI Alignment Forum.

Rimsky, N. (2023b). Red-teaming language models via activation engineering. AI Alignment Forum.

Rimsky, N. (2023c). Reducing sycophancy and improving honesty via activation steering. AI Alignment Forum.

Roemmele, M., Bejan, C. A., & Gordon, A. S. (2011). Choice of plausible alternatives: An evaluation of commonsense causal reasoning. *AAAI Spring Symposium*.

Roose, K. (2023). Bing's AI chat: "I want to be alive." *The New York Times*, 16.

Schramowski, P., Turan, C., Jentzsch, S., Rothkopf, C., & Kersting, K. (2019). BERT has a moral compass: Improvements of ethical and moral values of machines. *arXiv:1912.05238*.

Schwarz, N., & Clore, G. (1983). Mood, misattribution, and judgments of well-being: Informative and directive functions of affective states. *Journal of Personality and Social Psychology, 45*, 513–523.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*, 618–626.

Shaik, T., Tao, X., Xie, H., Li, L., Zhu, X., & Li, Q. (2023). Exploring the landscape of machine unlearning: A comprehensive survey and taxonomy. *arXiv:2305.06360*.

Shao, S., Ziser, Y., & Cohen, S. B. (2023). Gold doesn't always glitter: Spectral removal of linear and nonlinear guarded attribute information. *arXiv:2203.07893*.

Shen, Y., Gu, J., Tang, X., & Zhou, B. (2020). Interpreting the latent space of GANs for semantic face editing. *CVPR*, 9243–9252.

Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional networks: Visualising image classification models and saliency maps. *arXiv:1312.6034*.

Skjuve, M., Følstad, A., Fostervold, K. I., & Brandtzaeg, P. B. (2021). My chatbot companion — A study of human-chatbot relationships. *International Journal of Human-Computer Studies, 149*, 102601.

Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). SmoothGrad: Removing noise by adding noise. *arXiv:1706.03825*.

Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014). Striving for simplicity: The all convolutional net. *arXiv:1412.6806*.

Stanovsky, G., Smith, N. A., & Zettlemoyer, L. (2019). Evaluating gender bias in machine translation. *arXiv:1906.00591*.

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*, 3319–3328.

Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2013). Intriguing properties of neural networks. *arXiv:1312.6199*.

Talmor, A., Herzig, J., Lourie, N., & Berant, J. (2019). CommonsenseQA: A question answering challenge targeting commonsense knowledge. *NAACL-HLT*, 4149–4158.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., & Hashimoto, T. B. (2023). Stanford Alpaca: An instruction-following LLaMA model.

Thornley, E. (2023). There are no coherence theorems. AI Alignment Forum.

Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., & Manning, C. D. (2023). Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. *arXiv:2305.14975*.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv:2307.09288*.

Turner, A., M., M., Udell, D., Thiergart, L., & Mini, U. (2023a). Behavioural statistics for a maze-solving agent. AI Alignment Forum.

Turner, A., M., M., Udell, D., Thiergart, L., & Mini, U. (2023b). Maze-solving agents: Add a top-right vector, make the agent go to the top-right. AI Alignment Forum.

Turner, A., M., M., Udell, D., Thiergart, L., & Mini, U. (2023c). Steering GPT-2-XL by adding an activation vector. AI Alignment Forum.

Turner, A., M., M., Udell, D., Thiergart, L., & Mini, U. (2023d). Understanding and controlling a maze-solving policy network. AI Alignment Forum.

Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023e). Activation addition: Steering language models without optimization. *arXiv:2308.10248*.

Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023). Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. *arXiv:2305.04388*.

Upchurch, P., Gardner, J., Pleiss, G., Pless, R., Snavely, N., Bala, K., & Weinberger, K. (2017). Deep feature interpolation for image content changes. *CVPR*, 7064–7073.

Veit, A., Wilber, M. J., & Belongie, S. (2016). Residual networks behave like ensembles of relatively shallow networks. *NeurIPS 29*.

Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. R. (2018). GLUE: A multi-task benchmark and analysis platform for natural language understanding. *arXiv:1804.07461*.

Wang, K. R., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2023a). Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. *ICLR*.

Wang, Z., Gui, L., Negrea, J., & Veitch, V. (2023b). Concept algebra for (score-based) text-controlled generative models. *arXiv:2302.03693*.

Yosinski, J., Clune, J., Nguyen, A., Fuchs, T., & Lipson, H. (2015). Understanding neural networks through deep visualization. *arXiv:1506.06579*.

Zack, T., Lehman, E., Suzgun, M., Rodriguez, J. A., Celi, L. A., Gichoya, J., Jurafsky, D., Szolovits, P., Bates, D. W., Abdulnour, R.-E. E., Butte, A. J., & Alsentzer, E. (2023). Coding inequity: Assessing GPT-4's potential for perpetuating racial and gender biases in healthcare. *medRxiv*.

Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV*, 818–833.

Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K.-W. (2018). Gender bias in coreference resolution: Evaluation and debiasing methods. *arXiv:1804.06876*.

Zhong, Z., Wu, Z., Manning, C. D., Potts, C., & Chen, D. (2023). MQuAKE: Assessing knowledge editing in language models via multi-hop questions. *arXiv:2305.14795*.

Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). Learning deep features for discriminative localization. *CVPR*, 2921–2929.

Zhou, B., Sun, Y., Bau, D., & Torralba, A. (2018). Interpretable basis decomposition for visual explanation. *ECCV*, 119–134.

Zou, A., Wang, Z., Kolter, J. Z., & Fredrikson, M. (2023). Universal and transferable adversarial attacks on aligned language models. *arXiv:2307.15043*.

---

*Note: this document was produced by extracting and restructuring the text of the original PDF (`2310.01405v4.pdf`) for personal study purposes. Figures/diagrams from the original paper are described in words where relevant but not reproduced as images — refer back to the PDF or the arXiv page for the original visuals.*
