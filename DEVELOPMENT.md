## Setup
```bash
pip install -r requirements.txt
```

## Key Components
1. `deduplicate.py` - Fuzzy/MinHash deduplication
2. `categorize.py` - BERT/kw-based classification
3. `app.py` - Streamlit interface

## Testing
Run validation suite:
```bash
python tests/validate_pipeline.py
```
