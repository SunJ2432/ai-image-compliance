# AI Image Compliance Detection — Development Context

## Positioning
Help a content publisher acting as a Deployer under the EU AI Act assess whether an image is AI-generated, whether it constitutes a deepfake under Article 50(4), and what disclosure may be required. This prototype does not provide legal advice.

## Pipeline
Detection → VLM image description → sub-classification → regulatory retrieval (RAG) → LLM report generation → citation verification.

## Tech stack
Python package (`src/imgcompliance`), GitHub, Google Colab, pytest; planned: Hugging Face models, FAISS/LangChain retrieval, Pydantic, LangGraph. All execution happens in Colab notebooks; core logic belongs in Python modules.

## Design conventions
- Never identify or guess the identity of people in images.
- Artistic or satirical purpose is declared by the publisher, never inferred from image appearance.
- Article 50(2) is a Provider marking obligation; the planned C2PA check maps to this upstream signal, not a definitive legal compliance determination.
- Low confidence or failed citation verification must yield `needs_human_review`.
- Orchestration uses a deterministic LangGraph workflow, not an autonomous agent.
- Do not place core business logic in notebooks.
- Use genuine, traceable regulatory citations; do not fabricate legal provisions.
- Keep tokens in Colab Secrets; never commit them.

## Current progress

| Step | Description | Status |
|---|---|---|
| A0 | Repository skeleton | not started |
| A1 | Sample data | not started |
| A2 | Detection model | not started |
| A3 | Description and sub-classification | not started |
| A4 | Regulatory retrieval | not started |
| A5 | Report and verification | not started |
| A6 | End-to-end demo | not started |
| B1 | Full dataset | not started |
| B2 | Training and evaluation | not started |
| B3 | C2PA | not started |
| B4 | Classification and retrieval evaluation | not started |
| B5 | LoRA | not started |
| B6 | API and audit | not started |
| B7 | CI and Docker | not started |
| B8 | Wrap-up | not started |

Update this table after verifying each milestone.
