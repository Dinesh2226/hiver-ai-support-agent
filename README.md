# AI Support Agent for Customer Service on Twitter

An AI customer-support agent built for the Hiver SDE Intern take-home assignment using the Kaggle **Customer Support on Twitter** dataset.

The system is designed to:

1. Classify incoming customer messages into a small set of support intents.
2. Retrieve historically similar AmazonHelp conversations.
3. Draft a support reply grounded in historical responses.
4. Decide whether the case can be auto-handled or should be escalated to a human.
5. Evaluate classification, retrieval, escalation, and reply quality using held-out data and an LLM-as-judge.

---

## 1. Executive Summary

This project implements an AI support agent for Amazon customer-support messages from the Kaggle **Customer Support on Twitter** dataset.

The selected brand is **AmazonHelp**.

The system combines:

- Majority-class baseline
- TF-IDF + Logistic Regression baseline
- Qwen3:8b through Ollama
- TF-IDF historical retrieval
- Conservative deterministic escalation rules
- Evidence-constrained reply generation
- LLM-as-judge evaluation
- Human-reviewed reply evaluation

On a **200-example held-out golden set**, the intent-classification results were:

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority-class baseline | 37.0% | 5.4% | 20.0% |
| TF-IDF + Logistic Regression | 52.0% | 32.3% | 51.6% |
| Qwen3:8b | **56.0%** | **45.9%** | **57.1%** |

Compared with the TF-IDF baseline, Qwen3:8b improved by:

- **+4.0 percentage points** accuracy
- **+13.6 percentage points** macro F1
- **+5.5 percentage points** weighted F1

The main weakness is currently the **reply-generation stage** rather than intent classification.

On a stratified 30-example reply sample:

- Mean overall LLM-judge score: **1.87/5**
- Critical errors: **21/30**

This indicates that the prototype has a functional end-to-end workflow, but reply grounding and safe generation remain the main areas for improvement.

---

# 2. Problem Framing

Customer-support messages are difficult to automate reliably because a single message can contain:

- a concrete operational issue,
- strong emotional language,
- incomplete information,
- multiple issues,
- multilingual text,
- very short follow-up messages.

For example:

> "Why am I paying for Prime when you can't get my packages here on time?"

This contains Prime-related language and a concrete delivery problem.

The system therefore follows the principle:

> **The underlying operational issue takes precedence over sentiment or tone.**

The overall architecture is:

```text
                    Customer Message
                          |
                          v
                 Intent Classification
                          |
                          v
                 Historical Retrieval
                          |
              +-----------+-----------+
              |                       |
              v                       v
       Escalation Decision      Reply Generation
              |                       |
              +-----------+-----------+
                          |
                          v
                    Final Action