# doc-ai-invoice-extractor
Intelligent Document AI system for multilingual invoice field extraction (Convolve 4.0 GenAI Track)

This repository contains the hackathon-ready project under [doc_ai_project/README.md](doc_ai_project/README.md).

Quick start:

```bash
cd doc_ai_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python executable.py --pdf /path/to/invoice.pdf
streamlit run app.py
```

Robustness stress testing (deterministic noisy variants + report): see [doc_ai_project/README.md](doc_ai_project/README.md) and run `python robustness_eval.py --input /path/to/invoice.pdf`.
