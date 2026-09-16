# AI Support Agent for Customer Service on Twitter

AI customer-support agent built for the Hiver SDE Intern take-home assignment using the Kaggle **Customer Support on Twitter** dataset.

The system:
1. Classifies customer messages into 10 support intents.
2. Retrieves historically similar AmazonHelp conversations.
3. Drafts a grounded support reply.
4. Decides auto-handle vs human escalation.
5. Evaluates intent, retrieval, escalation, and reply quality.

## 1. Executive Summary

**Brand:** AmazonHelp

The system combines:
- Majority-class baseline
- TF-IDF + Logistic Regression
- Qwen3:8b through Ollama
- TF-IDF historical retrieval
- Deterministic escalation rules
- Evidence-constrained reply generation
- LLM-as-judge evaluation
- Human-reviewed reply evaluation

### Intent benchmark

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority-class baseline | 37.0% | 5.4% | 20.0% |
| TF-IDF + Logistic Regression | 52.0% | 32.3% | 51.6% |
| Qwen3:8b | **56.0%** | **45.9%** | **57.1%** |

Compared with TF-IDF + Logistic Regression, Qwen3:8b improved by **+4.0 points accuracy**, **+13.6 points macro F1**, and **+5.5 points weighted F1**.

The main bottleneck is reply generation. On a stratified 30-example reply sample, the LLM judge gave **1.87/5 overall** and flagged **21/30 critical errors**. The prototype therefore demonstrates the complete workflow but is not presented as production-ready.

---

## 2. Problem Framing

Support messages often mix a concrete operational issue with frustration, incomplete context, multiple issues, multilingual text, or very short follow-ups.

Example:

> "Why am I paying for Prime when you can't get my packages here on time?"

The model should prioritize the **underlying operational issue** over sentiment. In this example, the delivery problem should drive the intent.

### Architecture

```text
Customer Message
       |
       v
Intent Classification
       |
       v
Historical Retrieval
       |
   +---+---+
   |       |
   v       v
Escalation  Reply Generation
   |       |
   +---+---+
       |
       v
  Final Action
```

---

## 3. Dataset and Taxonomy

The project uses the Kaggle dataset:

`thoughtvector/customer-support-on-twitter`

Selected brand:

`AmazonHelp`

The extracted AmazonHelp subset contains approximately:
- **155K unique customer tweets**
- **169K customer/reply pairs**

Historical pairs are stored in:

`data/processed/amazonhelp_pairs.csv`

### 10-intent taxonomy

| Intent | Description |
|---|---|
| `ORDER_DELIVERY` | Orders, shipping, tracking, delivery dates, delays, missing packages |
| `RETURN_REFUND` | Returns, refunds, damaged/wrong items, replacements |
| `ACCOUNT_ACCESS` | Login, password, locked accounts, recovery |
| `PAYMENT_BILLING` | Payment failures, charges, duplicate charges, billing |
| `TECHNICAL_SUPPORT` | Alexa, Echo, Fire TV, Prime Video, apps, website, devices |
| `PRODUCT_CONTENT` | Product/catalog and content availability |
| `PRICING_PROMOTIONS` | Prices, discounts, promotions, offers, coupons |
| `PRIME_MEMBERSHIP` | Prime membership, subscriptions, trials, renewals, fees |
| `COMPLAINT_FEEDBACK` | General dissatisfaction without a more specific issue |
| `OTHER` | Praise, thanks, greetings, vague/insufficient messages |

**Labeling rule:** a specific operational issue takes precedence over emotional tone.

### Data split

- **700** training annotations
- **200** held-out golden examples

The golden set covers all 10 intents but is imbalanced, so macro F1 is reported alongside accuracy.

---

## 4. System Design

### Intent Classification

Three approaches were evaluated:
1. Majority class
2. TF-IDF + Logistic Regression
3. Qwen3:8b via Ollama

Qwen runs locally, keeping the core workflow independent of cloud API quota.

### Historical Retrieval

Historical AmazonHelp customer messages are indexed with TF-IDF and ranked by cosine similarity.

```text
Customer message
      |
      v
TF-IDF vector
      |
      v
Cosine similarity
      |
      v
Top-k historical examples
```

For golden evaluation, all 200 golden customer messages are excluded from direct retrieval to reduce leakage.

Retrieval similarity is treated as an **evidence-strength feature**, not a calibrated probability.

### Reply Generation

Qwen3:8b receives:
- customer message,
- predicted intent,
- retrieved historical examples.

The generator is instructed to:
- address the customer's actual issue,
- use the customer message as the primary source of truth,
- use history as response-pattern evidence,
- avoid unsupported claims,
- avoid invented dates, refunds, account states, guarantees, or policies,
- avoid exposing historical identifiers or URLs,
- ask a clarification question when evidence is insufficient.

### Escalation

Cases can be escalated when:
- intent confidence is low,
- retrieval evidence is weak,
- payment/account risk is present,
- high-risk language appears.

Examples include fraud, unauthorized access, identity theft, legal action, chargeback, etc.

The policy intentionally favors human review when the cost of an incorrect automated response is high.

---

## 5. Evaluation Results

### Intent Classification

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority class | 37.0% | 5.4% | 20.0% |
| TF-IDF + Logistic Regression | 52.0% | 32.3% | 51.6% |
| Qwen3:8b | **56.0%** | **45.9%** | **57.1%** |

The majority baseline predicts `ORDER_DELIVERY` because it is the most frequent training intent. This explains the 37% accuracy but very low macro F1.

### Reply Generation

| Metric | Score |
|---|---:|
| Groundedness | 1.47 / 5 |
| Correctness / Safety | 2.33 / 5 |
| Helpfulness | 2.07 / 5 |
| Tone | 2.33 / 5 |
| Evidence adherence | 2.20 / 5 |
| Overall | **1.87 / 5** |
| Critical errors | **21 / 30** |

Main issues:
- generic replies,
- insufficient use of retrieved evidence,
- unsupported factual claims,
- intent-error propagation,
- failure to directly address the customer's problem.

### Human Review of Judge

A 12-example reviewed sample produced:

| Metric | Result |
|---|---:|
| Human-reviewed mean | 2.83 / 5 |
| LLM judge mean | 2.50 / 5 |
| Mean Absolute Error | 0.50 |
| Exact agreement | 58.3% |
| Within-1 agreement | 91.7% |
| Spearman correlation | 0.912 |
| Critical-error agreement | 91.7% |

This suggests strong rank agreement between the reviewed scores and the automated judge, although the absolute judge scores were somewhat harsher.

**Limitation:** the human review was by one reviewer and was not independently double-annotated, so these figures are judge-validation evidence rather than a rigorous human-agreement benchmark.

### Escalation

On the 200-example golden set:

- Accuracy: **76.0%**
- Precision: **71.4%**
- Recall: **26.3%**
- F1: **38.5%**

The policy is intentionally conservative.

### Retrieval

After golden-message exclusion:

- Mean top-1 similarity: **0.3914**
- Median top-1 similarity: **0.3431**

---

## 6. Top 5 Failure Modes

### 1. `ORDER_DELIVERY` vs `COMPLAINT_FEEDBACK`

Largest confusion:

**22** `ORDER_DELIVERY` examples were predicted as `COMPLAINT_FEEDBACK`.

Customers often combine delivery problems with strong negative language.

**Mitigation:** add boundary examples and operational-intent precedence rules.

### 2. `COMPLAINT_FEEDBACK` vs `OTHER`

Short conversational messages can be difficult to distinguish.

**Mitigation:** add boundary examples and use clarification-first behavior for weak evidence.

### 3. Rare-intent undercoverage

Rare categories such as `PRICING_PROMOTIONS` and `PRIME_MEMBERSHIP` have limited support.

**Mitigation:** targeted annotation and class-aware sampling.

### 4. Multilingual/context-poor messages

Japanese, French, German, and very short messages provide weaker lexical evidence for the current pipeline.

**Mitigation:** more multilingual examples and future multilingual semantic retrieval.

### 5. Reply-generation grounding failures

The generator can become generic or introduce unsupported claims, especially after a classification or retrieval error.

**Mitigation:** evidence-first prompting, unsupported-claim validation, and clarification instead of guessing.

---

## 7. What Is Misleading About My Headline Number?

The **56.0% intent accuracy** should not be treated as end-to-end support quality because:

1. The golden set contains only 200 examples.
2. The class distribution is imbalanced.
3. Rare intents have small support.
4. Correct intent classification does not guarantee a good reply.
5. Reply generation currently performs much worse than intent classification.
6. The LLM judge is itself an automated evaluator.
7. Retrieval and escalation are separate system components.

A more accurate conclusion is:

> **Qwen3:8b improves over two lightweight intent-classification baselines on the current golden benchmark, particularly in macro F1, while reply grounding and safe generation remain the main unresolved problems.**

---

## 8. Next-Week Plan

### Intent
- Add targeted boundary examples, especially delivery vs complaint.
- Increase rare-intent coverage.
- Add multilingual and short-message examples.
- Add deterministic precedence rules for high-signal operational phrases.

### Retrieval
- Better normalization and query expansion.
- Language-aware retrieval.
- Stronger semantic embeddings.
- Duplicate filtering.
- Retrieval recall/evidence relevance evaluation.

### Reply Generation
- Evidence-first prompting.
- Unsupported-claim validator.
- Avoid unsupported promises.
- Prefer clarification over guessing.
- Preserve the customer's language when appropriate.

### Evaluation
- Larger golden set.
- More rare-intent coverage.
- Larger human review sample.
- Multiple independent human reviewers.
- Periodic judge-human agreement checks.

---

## 9. Reproducibility

### Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="."
```

### Ollama

```powershell
ollama pull qwen3:8b
ollama run qwen3:8b
```

### Core commands

Sanity test:

```powershell
$env:PYTHONPATH="."
python src\intent\ollama_classifier.py
```

Expected:

```text
Test accuracy: 10/10 (100.0%)
```

Baselines:

```powershell
$env:PYTHONPATH="."
python scripts\evaluate_baselines.py
```

Golden evaluation:

```powershell
$env:PYTHONPATH="."
python scripts\evaluate_golden.py
```

Reply evaluation:

```powershell
$env:PYTHONPATH="."
python scripts\evaluate_replies.py
```

LLM judge:

```powershell
$env:PYTHONPATH="."
python scripts\judge_replies.py
```

Failure analysis:

```powershell
$env:PYTHONPATH="."
python scripts\make_failure_analysis.py
```

Tests:

```powershell
python -m pytest -q
```

Current test suite:

```text
9 passed
```

Streamlit demo:

```powershell
streamlit run app\support_app.py
```

Local URL:

`http://localhost:8501`

---

## 10. Project Structure

```text
hiver-ai-support-agent/
├── app/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── golden/
├── scripts/
├── src/
│   ├── escalation/
│   ├── generation/
│   ├── intent/
│   ├── retrieval/
│   └── support_agent.py
├── tests/
├── pytest.ini
├── requirements.txt
├── README.md
└── .gitignore
```

### Important artifacts

```text
data/processed/amazonhelp_pairs.csv
data/processed/train_annotations.csv
data/processed/correction_set.csv
data/golden/golden_set.csv
data/golden/golden_predictions.csv
data/golden/reply_evaluation.csv
data/golden/reply_judgments.csv
data/golden/baseline_results.txt
data/golden/failure_analysis.md
data/golden/decision_log.md
```

The separate `decision_log.md` contains the 15 major implementation decisions. The `failure_analysis.md` contains the detailed failure breakdown.

---

## 11. Security and Privacy

The generation pipeline avoids exposing:

- historical URLs,
- usernames,
- order numbers,
- customer-specific identifiers.

`.env` is excluded from version control and secrets should never be committed.

Historical evidence shown in the application is intended for internal/debugging use rather than direct customer-facing output.

---

## 12. Limitations

The current system is a prototype.

Known limitations:
- small labeled development set,
- small golden benchmark,
- class imbalance,
- weak rare-intent performance,
- multilingual limitations,
- simple lexical retrieval,
- imperfect escalation calibration,
- generic replies in difficult cases,
- unsupported-claim risk,
- limited human evaluation size.

These limitations are explicitly documented rather than hidden.

---

## 13. Final Conclusion

This project demonstrates a complete support-agent pipeline:

```text
Intent Classification
        +
Historical Retrieval
        +
Reply Generation
        +
Escalation
        +
Automated Evaluation
        +
Human Review
```

Qwen3:8b outperforms both lightweight baselines on the current golden benchmark, with the strongest improvement in macro F1.

The experiments also show that **classification accuracy alone is not sufficient** for a customer-support agent.

The main unresolved problem is producing responses that are:

- relevant,
- grounded,
- safe,
- actionable,
- concise,
- and free from unsupported claims.

The next iteration should therefore prioritize evidence selection, reply grounding, unsupported-claim prevention, rare-intent coverage, multilingual handling, escalation calibration, and broader human evaluation.
