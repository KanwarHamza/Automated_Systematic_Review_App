# **Model Card: A fully Automated Systematic Review Application**

## **Model Overview**
A systematic review screening tool that classifies research papers as "Relevant" or "Not Relevant" using NLP and ML techniques. Designed to accelerate literature reviews by prioritizing papers through a 6-stage pipeline.

**Key Features**:
- API integration (PubMed, arXiv, Crossref)
- Fuzzy/ML-based deduplication
- Multi-level categorization (keyword + BERT)
- Export-ready analysis

## **Usage**
```python
# Example: Loading the deduplication model
import joblib
model = joblib.load("deduplication_model.pkl")  # Fuzzy matching + MinHash
df_clean = model.deduplicate(df, columns=['Title', 'Abstract'])
```

**Input**:  
- Research paper metadata (Title, Abstract, DOI, etc.)  
- Either via file upload or API queries  

**Output**:  
- Categorized papers with:  
  - Primary/secondary categories  
  - Deduplication flags  
  - Analysis dimensions  

## **System Architecture**
| Component | Technology |
|-----------|------------|
| Core Engine | Python (scikit-learn, transformers) |
| Deduplication | FuzzyWuzzy + MinHashLSH |
| Categorization | Keyword matching + BERT embeddings |
| UI | Streamlit |

**Dependencies**:  
- Requires Python 3.8+  
- GPU recommended for BERT models  

## **Implementation Requirements**
| Task | Hardware | Time Estimate |
|------|----------|---------------|
| Deduplication (10K papers) | CPU | 2-5 mins |
| BERT Categorization | GPU (T4) | 1 min/batch |
| Full Pipeline | 4GB RAM | 10-30 mins |

## **Model Characteristics**
- **Initialization**: Hybrid approach
  - Exact/fuzzy matching for deduplication
  - Fine-tuned BERT (`all-MiniLM-L6-v2`) for semantic categorization
- **Size**:  
  - 80MB (BERT model)  
  - <1MB (scikit-learn models)  

## **Data Overview**
**Training Data**:  
- No fixed dataset - adapts to user's input  
- Sample structure:  
  ```csv
  Title,Abstract,DOI,Year,Source
  ```

**Evaluation**:  
- Precision/recall tested on synthetic datasets  
- 95%+ accuracy on exact duplicates  
- 85-90% on fuzzy matches  

## **Evaluation Results**
| Metric | Performance |
|--------|-------------|
| Duplicate Detection (F1) | 0.92 |
| Primary Categorization Accuracy | 0.88 |
| BERT Cluster Purity | 0.79 |

**Known Limitations**:  
- Performance drops with:  
  - Highly technical jargon  
  - Non-English abstracts  
  - Incomplete metadata  

## **Ethical Considerations**
**Risks**:  
- May inherit biases from training corpora  
- Over-reliance could miss novel research  

**Mitigations**:  
- Human-in-the-loop validation  
- Clear "Uncategorized" buckets  
- Confidence thresholds  

## **License**
- **Code**: Apache 2.0  
- **User Data**: Remains property of uploader  

**Developed By**: Kanwar Hamza Shuja  
**Contact**: kanwarhamzashuja@gmail.com  

## **DOI**
- [[DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15375377.svg)](https://doi.org/10.5281/zenodo.15375377)

