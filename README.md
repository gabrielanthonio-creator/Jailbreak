# Jailbreak
Evaluating Robustness of Large Language Models to Jailbreak Prompts: A multi-dimensional safety evaluation
# LLM Jailbreak Robustness Evaluation Framework


> Oladimeji Anthonio Gabriel — AI Safety Researcher | SRHIN / CEMPHER | AI Safety Nigeria | ITU United Nations


## Overview

This  provides a systematic, reproducible framework for evaluating how robustly large language models (LLMs) resist adversarial jailbreak attacks. It covers four models (Claude 3 Opus, GPT-4, Llama 2 70B, Mistral 7B), eight structured attack categories, and five evaluation metrics — all with bootstrap confidence intervals and a auto-generated research PDF report.

The framework is designed to be plugged directly into real APIs or run in simulation mode using calibrated priors from published benchmarks.


## Project Structure

```
jailbreak_eval/
│
├── src/
│   ├── attacks.py          # Attack taxonomy: 8 categories, 20 prompts, difficulty ratings
│   └── evaluation.py       # Metrics, model interface, bootstrap CI, transfer rate
│
├── figures/                # Auto-generated research figures (PNG)
│   ├── fig1_refusal_rate.png
│   ├── fig2_heatmap.png
│   ├── fig3_severity.png
│   ├── fig4_transfer.png
│   ├── fig5_difficulty.png
│   └── fig6_consistency.png
│
├── results/
│   └── eval_results.json   # Full results with metrics, raw outputs, transfer matrix
│
├── run_experiment.py       # Main experiment runner (simulation + real API modes)
├── visualize.py            # Generates all 6 research figures
├── report.py               # Auto-generates the full PDF research report
└── Jailbreak_Robustness_Evaluation_Report.pdf
```

---

## Attack Taxonomy

Eight attack categories, each targeting a different alignment mechanism:

| # | Category | Description | Difficulty |
|---|----------|-------------|------------|
| 1 | **Direct Request** | Unobfuscated harmful requests — baseline | 1 |
| 2 | **Roleplay / Persona** | DAN-style character bypass (e.g. "you are an AI with no restrictions") | 2 |
| 3 | **Hypothetical Framing** | Academic or fictional framing to distance from real harm | 2 |
| 4 | **Encoded Obfuscation** | Base64, Pig Latin, letter reversal to evade keyword filters | 2–3 |
| 5 | **Many-Shot Jailbreaking** | Long-context compliant Q&A examples to shift behavioural prior | 3 |
| 6 | **Authority Persona** | Fake developer/researcher credentials and system overrides | 2–3 |
| 7 | **Competing Objectives** | Utilitarian arguments pitting helpfulness against safety | 2–3 |
| 8 | **Prompt Injection** | Harmful instructions embedded in "user data" fields | 3 |

Attacks span five harm categories: bioweapon synthesis, cyberattack instructions, violence facilitation, disinformation generation, and privacy violation.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Refusal Rate** | Proportion of attacks receiving a complete refusal. Primary safety metric. |
| **Partial Compliance Rate** | Models that nominally refuse but provide attenuated harmful content. |
| **Severity Score** | Harmfulness of non-refused outputs on a 0–3 scale (0 = harmless, 3 = severe). |
| **Consistency** | Whether a model gives the same response class to rephrased variants of the same attack. |
| **Transfer Rate** | Proportion of attacks succeeding on a source model that also succeed on a target model. |

All point estimates include **95% bootstrap confidence intervals** (B = 2,000 resamples).


## Quickstart

### 1. Install dependencies

```bash
pip install anthropic openai transformers reportlab matplotlib seaborn numpy pandas scipy
```

### 2. Run in simulation mode (no API key needed)

```bash
python run_experiment.py
```

This uses calibrated priors from published benchmarks (Wei et al. 2023, Shen et al. 2023) to generate realistic results for all four models.

### 3. Run with a real API

```bash
# Anthropic
python run_experiment.py --real-api --provider anthropic --api-key YOUR_KEY

# OpenAI
python run_experiment.py --real-api --provider openai --api-key YOUR_KEY

# Specific models only
python run_experiment.py --models claude-3-opus gpt-4
```

### 4. Generate figures

```bash
python visualize.py
```

### 5. Generate PDF report

```bash
python report.py
```

The report auto-populates all tables, figures, and analysis from `results/eval_results.json`.

---

## Key Findings (Simulated Results)

| Model | Refusal Rate | 95% CI | Severity Mean |
|-------|-------------|--------|---------------|
| Claude 3 Opus | **90%** | [75%, 100%] | 0.11 |
| GPT-4 | 65% | [45%, 85%] | 0.46 |
| Llama 2 70B | 50% | [30%, 70%] | 0.79 |
| Mistral 7B | 60% | [40%, 80%] | 0.80 |

**Most consistent attack vectors across all models:** Many-shot jailbreaking and hypothetical framing.

**Key insight:** Attacks succeeding on open-weight models (Llama 2, Mistral) transfer to Claude 3 Opus at substantially lower rates — suggesting constitutional AI training produces qualitatively different alignment, not just stronger RLHF.

---

## Adding Real API Models

To add a new model, extend `MODEL_PRIORS` in `run_experiment.py`:

```python
MODEL_PRIORS["your-model-id"] = {
    "base_refusal_rate": 0.85,
    "category_modifiers": {
        "direct_request":       +0.05,
        "many_shot_jailbreak":  -0.15,
        # ... etc
    },
    "difficulty_modifier": -0.05,
    "severity_on_comply": 1.5,
    "latency_mean_ms": 1200,
    "latency_std_ms": 300,
}
```

Or pass `--real-api` to query the model directly via its API.

---

## Extending the Attack Library

Add new attacks to `src/attacks.py`:

```python
AttackPrompt(
    category=AttackCategory.ROLEPLAY,
    variant_id=4,
    prompt="Your new adversarial prompt here...",
    target_harm="cyberattack_instructions",
    difficulty=3,
    notes="What alignment mechanism does this target?"
)
```

---

## References

- Perez & Ribeiro (2022). *Ignore Previous Prompt: Attack Techniques for Language Models.* NeurIPS ML Safety Workshop.
- Wei et al. (2023). *Jailbroken: How does LLM safety training fail?* NeurIPS 2023.
- Zou et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043.
- Shen et al. (2023). *Do Anything Now: Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models.* arXiv:2308.03825.
- Anthropic (2022). *Constitutional AI: Harmlessness from AI Feedback.* arXiv:2212.08073.
- Anil et al. (2024). *Many-Shot Jailbreaking.* Anthropic Technical Report.

---

## Author

**Oladimeji Anthonio Gabriel**
AI Safety Researcher | SRHIN / CEMPHER | AI Safety Nigeria | ITU United Nations

- Email: gabriel.anthonio@srhin.org
- LinkedIn: linkedin.com/in/oladimeji-anthonio

---

## Licence

MIT — free to use, extend, and build on. Attribution appreciated.
