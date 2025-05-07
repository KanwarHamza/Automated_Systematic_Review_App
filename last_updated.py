import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib
import zipfile
from io import BytesIO
import matplotlib.pyplot as plt
import pyarrow as pa
import torch
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from fuzzywuzzy import fuzz
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Fuzzy matching imports
from rapidfuzz import process, fuzz as rapidfuzz
from fuzzywuzzy import fuzz as fuzzywuzzy_fuzz

# Deduplication imports
from datasketch import MinHash, MinHashLSH

# Transformers imports
from transformers import BertTokenizer, BertModel
from transformers import pipeline as transformers_pipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Constants
KEEP_OPTION_MAP = {
    'First occurrence': 'first',
    'Last occurrence': 'last',
    'None (remove all)': False
}

# Initialize session state
def init_session_state():
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 1
        st.session_state.raw_data = None
        st.session_state.processed_data = None
        st.session_state.unmatched_data = None
        st.session_state.summaries = []
        st.session_state.categorization_results = {
            'matched': None,
            'unmatched': None,
            'show_results': False
        }
        st.session_state.secondary_categorization_results = {
            'matched': pd.DataFrame(),
            'unmatched': pd.DataFrame(),
            'show_results': False
        }
        st.session_state.bert_model_loaded = False
        st.session_state.bert_tokenizer = None
        st.session_state.bert_model = None
        st.session_state.analysis_results = None
        st.session_state.eda_report = None

init_session_state()

def reset_pipeline():
    st.session_state.current_stage = 1
    st.session_state.raw_data = None
    st.session_state.processed_data = None
    st.session_state.unmatched_data = None
    st.session_state.summaries = []
    st.session_state.categorization_results = {
        'matched': None,
        'unmatched': None,
        'show_results': False
    }
    st.session_state.secondary_categorization_results = {
        'matched': pd.DataFrame(),
        'unmatched': pd.DataFrame(),
        'show_results': False
    }
    st.session_state.analysis_results = None
    st.session_state.eda_report = None
    st.rerun()

# API Fetch Functions
def fetch_from_pubmed(query, max_results=100):
    """Fetch articles from PubMed API with robust error handling"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}"
    
    try:
        # Fetch article IDs
        response = requests.get(search_url)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        id_list = [id_.text for id_ in root.findall(".//Id") if id_.text]
        
        if not id_list:
            return pd.DataFrame()
        
        # Fetch article details in batches
        articles = []
        batch_size = 100  # PubMed's max for efetch
        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i+batch_size]
            fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={','.join(batch_ids)}&retmode=xml"
            
            try:
                response = requests.get(fetch_url)
                response.raise_for_status()
                root = ET.fromstring(response.text)
                
                for article in root.findall(".//PubmedArticle"):
                    try:
                        title = article.find(".//ArticleTitle")
                        title = title.text if title is not None else ""
                        
                        abstract = article.find(".//AbstractText")
                        abstract = abstract.text if abstract is not None else ""
                        
                        pub_date = article.find(".//PubDate")
                        year = ""
                        if pub_date is not None:
                            year_elem = pub_date.find("Year")
                            year = year_elem.text if year_elem is not None else ""
                        
                        doi = article.find(".//ArticleId[@IdType='doi']")
                        doi = doi.text if doi is not None else ""
                        
                        articles.append({
                            "Title": title.strip() if title else "",
                            "Abstract": abstract.strip() if abstract else "",
                            "Year": year,
                            "DOI": doi,
                            "Source": "PubMed"
                        })
                    except Exception as e:
                        continue
                        
            except Exception as e:
                continue
        
        return pd.DataFrame(articles)
    
    except Exception as e:
        st.error(f"PubMed API error: {str(e)}")
        return pd.DataFrame()

def fetch_from_arxiv(query, max_results=100):
    """Fetch articles from arXiv API with robust error handling"""
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        
        articles = []
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall("atom:entry", ns):
            try:
                title = entry.find("atom:title", ns)
                title = title.text.strip() if title is not None else ""
                
                summary = entry.find("atom:summary", ns)
                summary = summary.text.strip() if summary is not None else ""
                
                published = entry.find("atom:published", ns)
                year = ""
                if published is not None and published.text:
                    try:
                        year = datetime.strptime(published.text, "%Y-%m-%dT%H:%M:%SZ").year
                    except:
                        pass
                
                doi = entry.find("atom:doi", ns)
                doi = doi.text if doi is not None else ""
                
                articles.append({
                    "Title": title,
                    "Abstract": summary,
                    "Year": str(year) if year else "",
                    "DOI": doi,
                    "Source": "arXiv"
                })
            except Exception as e:
                continue
        
        return pd.DataFrame(articles)
    
    except Exception as e:
        st.error(f"arXiv API error: {str(e)}")
        return pd.DataFrame()

def fetch_from_crossref(query, max_results=100):
    """Fetch articles from Crossref API with robust error handling"""
    url = f"https://api.crossref.org/works?query={query}&rows={max_results}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        for item in data.get("message", {}).get("items", []):
            try:
                title = item.get("title", [""])[0] if item.get("title") else ""
                
                abstract = ""
                if "abstract" in item:
                    abstract = item["abstract"]
                elif "abstracts" in item:
                    abstract = item["abstracts"][0].get("text", "") if item["abstracts"] else ""
                
                year = ""
                if "created" in item and "date-parts" in item["created"]:
                    date_parts = item["created"]["date-parts"][0]
                    year = str(date_parts[0]) if date_parts else ""
                
                doi = item.get("DOI", "")
                
                articles.append({
                    "Title": title.strip() if title else "",
                    "Abstract": abstract.strip() if abstract else "",
                    "Year": year,
                    "DOI": doi,
                    "Source": "Crossref"
                })
            except Exception as e:
                continue
        
        return pd.DataFrame(articles)
    
    except Exception as e:
        st.error(f"Crossref API error: {str(e)}")
        return pd.DataFrame()

# Improved Deduplication Methods
def fuzzy_deduplicate(df, columns, threshold=90, keep_option='First occurrence', use_rapidfuzz=True):
    keep_method = KEEP_OPTION_MAP[keep_option]
    
    # Create a combined key from selected columns
    df['combined_key'] = df[columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    
    # Pre-process text: lowercase, remove punctuation, normalize whitespace
    df['processed_key'] = df['combined_key'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()
    
    duplicate_indices = set()
    texts = df['processed_key'].tolist()
    total = len(texts)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, text in enumerate(texts):
        if i in duplicate_indices:
            continue
            
        if use_rapidfuzz:
            matches = process.extract(
                text, 
                texts[i+1:], 
                scorer=rapidfuzz.WRatio,
                score_cutoff=threshold,
                limit=None,
                processor=None  # We already processed the text
            )
            for match, score, idx in matches:
                if score >= threshold:
                    duplicate_indices.add(i+1+idx)
        else:
            for j in range(i+1, total):
                if j in duplicate_indices:
                    continue
                score = fuzzywuzzy_fuzz.token_set_ratio(text, texts[j])
                if score >= threshold:
                    duplicate_indices.add(j)
        
        progress = (i + 1) / total
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Processed {i+1}/{total} documents (found {len(duplicate_indices)} duplicates)")
    
    progress_bar.empty()
    status_text.empty()
    
    if keep_method == 'first':
        df_clean = df[~df.index.isin(duplicate_indices)]
    elif keep_method == 'last':
        last_occurrences = df[~df.index.isin(duplicate_indices)].copy()
        df_clean = last_occurrences
    else:
        df_clean = df[~df.index.isin(duplicate_indices)]
    
    return df_clean.drop(columns=['combined_key', 'processed_key'])

def minhash_deduplicate(df, text_col='Title', threshold=0.85):
    minhashes = []
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    
    for idx, text in enumerate(df[text_col]):
        mh = MinHash(num_perm=128)
        for word in str(text).lower().split():
            mh.update(word.encode('utf-8'))
        lsh.insert(idx, mh)
        minhashes.append(mh)
    
    duplicate_indices = set()
    for idx in range(len(df)):
        if idx not in duplicate_indices:
            matches = lsh.query(minhashes[idx])
            duplicate_indices.update(matches)
    
    return df[~df.index.isin(duplicate_indices)]

# Save functionality
def add_save_option(stage_number):
    if st.session_state.processed_data is not None:
        with st.sidebar.expander("💾 Save Current Progress"):
            save_name = st.text_input(
                "File name:", 
                f"research_papers_stage_{stage_number}",
                key=f"save_stage_{stage_number}_name"
            )
            
            if st.button(f"Save Stage {stage_number} Results", key=f"save_stage_{stage_number}"):
                with st.spinner("Saving..."):
                    try:
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            st.session_state.processed_data.to_excel(writer, sheet_name='Processed_Papers', index=False)
                            if 'unmatched_data' in st.session_state and not st.session_state.unmatched_data.empty:
                                st.session_state.unmatched_data.to_excel(writer, sheet_name='Unmatched_Papers', index=False)
                        
                        buffer.seek(0)
                        st.download_button(
                            label="📥 Download Excel File",
                            data=buffer,
                            file_name=f"{save_name}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"download_stage_{stage_number}_excel"
                        )
                        st.success("File ready for download!")
                    except Exception as e:
                        st.error(f"Error saving file: {str(e)}")

def sanitize_sheet_name(name):
    """Clean sheet names for Excel output"""
    name = re.sub(r'[\\/*?:\[\]]', '_', name)
    return name[:31]

# Stage 1: Data Collection & EDA
def stage_data_eda():
    st.header("📊 Stage 1: Data Collection & Exploratory Analysis")
    
    with st.expander("📝 Instructions", expanded=True):
        st.markdown("""
        **Purpose**: Understand your dataset's characteristics before processing.
        
        **How to use**:
        1. Upload your dataset or fetch from APIs
        2. Generate the automated EDA report
        3. Review data quality, distributions, and patterns
        4. Identify potential issues before proceeding
        
        **Tips**:
        - Check the "Variables" section for column-specific insights
        - Review "Interactions" for potential correlations
        - Examine "Missing values" to identify data completeness
        """)

    # Data source selection
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "Upload my own file"
    
    st.session_state.data_source = st.radio(
        "Select data source:",
        ["Upload my own file", "Download from scientific API"],
        key="data_source_radio"
    )

    # API Download Section
    if st.session_state.data_source == "Download from scientific API":
        st.subheader("API Download Settings")
        
        if 'api_choice' not in st.session_state:
            st.session_state.api_choice = "PubMed"
            
        st.session_state.api_choice = st.selectbox(
            "Select API:",
            ["PubMed", "arXiv", "Crossref"],
            key="api_choice_select"
        )
        
        query = st.text_input(
            "Search query:",
            "",
            key="api_query_input",
            help="e.g., 'machine learning in healthcare'"
        )
        
        max_results = st.number_input(
            "Maximum results to fetch:",
            1, 1000, 100,
            key="api_max_results"
        )
        
        if st.button("Fetch Data from API", key="api_fetch_button"):
            with st.spinner(f"Downloading papers from {st.session_state.api_choice}..."):
                try:
                    if st.session_state.api_choice == "PubMed":
                        df = fetch_from_pubmed(query, max_results)
                    elif st.session_state.api_choice == "arXiv":
                        df = fetch_from_arxiv(query, max_results)
                    elif st.session_state.api_choice == "Crossref":
                        df = fetch_from_crossref(query, max_results)
                    
                    if df.empty:
                        st.warning("No results found for your query")
                    else:
                        st.session_state.raw_data = df
                        st.success(f"Downloaded {len(df)} records from {st.session_state.api_choice}")
                        
                        # Show quick preview
                        st.subheader("Data Preview")
                        st.dataframe(df.head())
                        
                except Exception as e:
                    st.error(f"Error downloading data: {str(e)}")

    # File Upload Section
    else:
        uploaded_file = st.file_uploader(
            "Upload your research papers (Excel/CSV)", 
            type=['xlsx', 'xls', 'csv'],
            key="stage1_uploader"
        )
        
        if uploaded_file:
            with st.spinner("Loading data..."):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.raw_data = df
                    st.success(f"Loaded {len(df)} records")
                    
                    # Show quick preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
                        
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")

    # EDA Report Generation
    if 'raw_data' in st.session_state and st.session_state.raw_data is not None:
        if st.button("Generate EDA Report", key="generate_eda"):
            with st.spinner("Creating comprehensive EDA report..."):
                try:
                    # Create profile report
                    profile = ProfileReport(
                        st.session_state.raw_data,
                        title="Research Papers Profiling Report",
                        explorative=True,
                        minimal=False,
                        correlations={
                            "pearson": {"calculate": True},
                            "spearman": {"calculate": True},
                            "kendall": {"calculate": True},
                            "phi_k": {"calculate": True},
                        },
                        interactions={
                            "continuous": True,
                        },
                        missing_diagrams={
                            'heatmap': True,
                            'dendrogram': True,
                        },
                    )
                    
                    # Save to HTML
                    profile.to_file("eda_report.html")
                    st.session_state.eda_report = "eda_report.html"
                    
                    # Display in Streamlit
                    st.subheader("Exploratory Data Analysis Report")
                    components.html(profile.to_html(), height=800, scrolling=True)
                    
                    # Download button
                    with open("eda_report.html", "rb") as f:
                        st.download_button(
                            label="📥 Download Full EDA Report",
                            data=f,
                            file_name="research_papers_eda.html",
                            mime="text/html"
                        )
                    
                    st.success("EDA report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating EDA report: {str(e)}")
                    st.error("Please ensure your dataset has at least 2 columns with valid data")

    if 'raw_data' in st.session_state and st.button("Continue to Stage 2", key="stage1_continue"):
        st.session_state.current_stage = 2
        st.rerun()

# Stage 2: Duplicate Removal
def stage_duplicate_removal():
    st.header("🔍 Stage 2: Duplicate Removal")
    
    with st.expander("📝 Instructions", expanded=True):
        st.markdown("""
        **Purpose**: Remove duplicate papers from your dataset.
        
        **How to use**:
        1. Select columns to check for duplicates (Title, DOI, etc.)
        2. Choose a deduplication method based on your needs
        3. Review the results before proceeding
        
        **Tips**:
        - For exact matches, use "Exact match (strict)"
        - For similar but not identical papers, use fuzzy matching
        - The combined method provides the best balance of precision and recall
        """)

    if st.session_state.raw_data is None:
        st.warning("Please complete Stage 1 first")
        return
    
    df = st.session_state.raw_data.copy()
    
    with st.sidebar:
        st.subheader("Duplicate Removal Settings")
        
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Columns to check for duplicates:",
            all_columns,
            default=['Title', 'DOI', 'Abstract'] if {'Title','DOI','Abstract'}.issubset(all_columns) 
            else all_columns[:3],
            key="stage2_columns"
        )
        
        keep_option_text = st.radio(
            "Which duplicates to keep:",
            options=list(KEEP_OPTION_MAP.keys()),
            index=0,
            key="stage2_keep_option"
        )
        
        with st.expander("Advanced Options"):
            dedupe_method = st.selectbox(
                "Deduplication method:",
                [
                    "Exact match (strict)", 
                    "Fuzzy match (conservative - 95%)",
                    "Fuzzy match (balanced - 90%)",
                    "Fuzzy match (aggressive - 85%)",
                    "Combined Method (recommended)"
                ],
                index=4,
                key="stage2_method"
            )
            
            if "Fuzzy" in dedupe_method:
                if "conservative" in dedupe_method:
                    default_threshold = 95
                elif "balanced" in dedupe_method:
                    default_threshold = 90
                else:
                    default_threshold = 85
                    
                similarity_threshold = st.slider(
                    "Similarity threshold", 
                    70, 100, 
                    default_threshold,
                    key="stage2_fuzzy_threshold"
                )
            
            if dedupe_method == "Exact match (strict)":
                ignore_na = st.checkbox("Ignore NA values", True, key="stage2_ignore_na")
                case_sensitive = st.checkbox("Case sensitive matching", False, key="stage2_case")
                
            if dedupe_method == "Combined Method (recommended)":
                st.info("""
                Uses a two-step approach:
                1. First pass with MinHash (85% similarity) for quick candidate finding
                2. Second pass with fuzzy matching (90% similarity) for precise verification
                """)
    
    if st.button("Remove Duplicates", key="stage2_remove_duplicates"):
        with st.spinner("Processing..."):
            df = st.session_state.raw_data.copy()
            original_count = len(df)
            
            # Apply selected method
            if dedupe_method == "Exact match (strict)":
                if ignore_na:
                    df = df.fillna('')
                if not case_sensitive:
                    for col in selected_columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.lower().str.strip()
                df_clean = df.drop_duplicates(
                    subset=selected_columns,
                    keep=KEEP_OPTION_MAP[keep_option_text]
                )
                method_used = "Strict exact matching"
                
            elif dedupe_method == "Combined Method (recommended)":
                # First pass with MinHash for quick candidate finding
                df_temp = minhash_deduplicate(df, text_col=selected_columns[0], threshold=0.85)
                
                # Second pass with fuzzy matching for verification
                df_clean = fuzzy_deduplicate(
                    df_temp, 
                    columns=selected_columns,
                    threshold=90,
                    keep_option=KEEP_OPTION_MAP[keep_option_text],
                    use_rapidfuzz=True
                )
                method_used = "Combined (MinHash + Fuzzy)"
                
            else:  # Fuzzy variants
                df_clean = fuzzy_deduplicate(
                    df,
                    columns=selected_columns,
                    threshold=similarity_threshold,
                    keep_option=KEEP_OPTION_MAP[keep_option_text],
                    use_rapidfuzz=True
                )
                method_used = f"Fuzzy matching ({similarity_threshold}%)"
            
            # Store results
            st.session_state.processed_data = df_clean
            st.session_state.summaries.append({
                'stage': 2,
                'action': 'Duplicate removal',
                'method': method_used,
                'original_count': original_count,
                'remaining_count': len(df_clean),
                'removed_count': original_count - len(df_clean)
            })
            
            # Show results
            st.success(f"Removed {original_count-len(df_clean)} duplicates. {len(df_clean)} records remain.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Method used:**", method_used)
                if len(df) - len(df_clean) > 0:
                    st.write(f"**Duplicate ratio:** {(original_count-len(df_clean))/original_count:.1%}")
            
            with col2:
                fig, ax = plt.subplots()
                ax.bar(['Original', 'After Deduplication'], [original_count, len(df_clean)])
                ax.set_title('Document Count Comparison')
                st.pyplot(fig)
            
            if len(df) - len(df_clean) > 0:
                with st.expander("Show sample of removed duplicates"):
                    st.dataframe(
                        df[df.duplicated(
                            subset=selected_columns,
                            keep=KEEP_OPTION_MAP[keep_option_text]
                        )].head()
                    )

    if 'processed_data' in st.session_state and st.button("Continue to Stage 3", key="stage2_continue"):
        st.session_state.current_stage = 3
        st.rerun()

# Stage 3: Primary Categorization
def stage_primary_categorization():
    st.header("📚 Stage 3: Primary Categorization")
    
    with st.expander("📝 Instructions", expanded=True):
        st.markdown("""
        **Purpose**: Categorize your research papers based on their content.
        
        **How to use**:
        1. Select which text columns to analyze (e.g., Title, Abstract)
        2. Choose a categorization method:
           - **Keyword Matching**: Match papers based on keywords you define
           - **BERT Semantic Matching**: Uses AI to group similar papers
        3. For Keyword Matching:
           - Enter your core categories (one per line)
           - Enter additional categories (one per line)
           - Define keyword variants for each category
           - Set matching threshold and strategy
        4. For BERT Matching:
           - Set number of categories
           - Adjust batch size and similarity threshold
        5. Click "Categorize Papers" to process your data
        
        **Tips**:
        - For keyword matching, include all possible variations of terms
        - Higher thresholds will result in more precise but fewer matches
        - BERT works best with GPU acceleration
        """)
        
    if st.session_state.processed_data is None:
        st.warning("Please complete Stage 2 first")
        return
    
    df = st.session_state.processed_data.copy()
    
    with st.sidebar:
        st.subheader("⚙️ Settings")
        method = st.radio(
            "Categorization Method:",
            ["Standalone Exact-Match", 
             "Configurable Keyword Matching", 
             "BERT Semantic Matching"],
            index=0
        )
        
        text_columns = st.multiselect(
            "Text columns to analyze:",
            df.select_dtypes(include='object').columns.tolist(),
            default=['Title', 'Abstract'] if {'Title','Abstract'}.issubset(df.columns) 
            else df.columns[:2]
        )
        
        # COMMON OUTPUT OPTIONS
        st.markdown("### Output Options")
        export_excel = st.checkbox("Export to Excel", True)
        show_details = st.checkbox("Show detailed results", True)

    # METHOD-SPECIFIC UI
    if method == "Standalone Exact-Match":
        with st.expander("🔍 Keyword Configuration", expanded=True):
            st.markdown("**Core Research Themes**")
            core_input = st.text_area(
                "Enter core categories (format: 'category: variant1, variant2')",
                "metacognition: metacognition, meta-cognition\n"
                "theory of mind: ToM, theory of mind"
            )
            
            st.markdown("**Additional Categories**")
            additional_input = st.text_area(
                "Enter additional categories (same format)",
                "Alcohol: alcohol, alcoholic\n"
                "Schizophrenia: schizophrenia"
            )

    elif method == "Configurable Keyword Matching":
        with st.expander("🔍 Fuzzy Matching Configuration", expanded=True):
            st.markdown("### Core Categories")
            core_categories = st.text_area(
                "Enter core categories (one per line):",
                "",
                height=100,
                key="stage2_core_categories",
                help="These represent your main research themes (e.g., 'machine learning', 'neuroscience')"
            ).split('\n')
            
            st.markdown("### Additional Categories")
            additional_categories = st.text_area(
                "Enter additional categories (one per line):",
                "",
                height=100,
                key="stage2_additional_categories",
                help="These will be combined with core categories (e.g., 'healthcare', 'finance')"
            ).split('\n')
            
            st.markdown("### Matching Options")
            min_score = st.slider("Minimum confidence threshold:", 0.0, 1.0, 0.5, key="stage3_min_score")
            strategy = st.radio("Matching strategy:", ["Highest score", "First match"], index=0, key="stage3_strategy")
        
    else:  # BERT Semantic Matching
        with st.expander("🧠 BERT Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                clustering_mode = st.radio(
                    "Clustering Approach:",
                    ["Pure Semantic", "Keyword-Guided"],
                    help="Keyword-guided uses your terms as cluster seeds"
                )
            with col2:
                num_clusters = st.number_input(
                    "Number of clusters:", 
                    min_value=2, max_value=20, value=5
                )
            
            if clustering_mode == "Keyword-Guided":
                seed_keywords = st.text_area(
                    "Seed keywords (one per line):",
                    "machine learning\nneuroscience\nclinical research"
                ).split('\n')

    if st.button("🚀 Categorize Papers"):
        with st.spinner("Processing..."):
            # PROCESS BASED ON SELECTED METHOD
            if method == "Standalone Exact-Match":
                # Parse keyword inputs
                core_keywords = {}
                for line in core_input.split('\n'):
                    if ':' in line:
                        cat, variants = line.split(':', 1)
                        core_keywords[cat.strip()] = [v.strip() for v in variants.split(',')]
                
                additional_keywords = {}
                for line in additional_input.split('\n'):
                    if ':' in line:
                        cat, variants = line.split(':', 1)
                        additional_keywords[cat.strip()] = [v.strip() for v in variants.split(',')]
                
                # Exact matching logic
                df['search_text'] = df[text_columns].apply(
                    lambda x: ' '.join(x.dropna().astype(str)).lower(), axis=1)
                
                results = []
                for _, row in df.iterrows():
                    matches = set()
                    # Check core keywords
                    for core, variants in core_keywords.items():
                        if any(f' {v.lower()} ' in f' {row["search_text"]} ' for v in variants):
                            matches.add(core)
                    # Check additional keywords
                    for add, variants in additional_keywords.items():
                        if any(f' {v.lower()} ' in f' {row["search_text"]} ' for v in variants):
                            matches.add(add)
                    
                    results.append(", ".join(sorted(matches)) if matches else "Uncategorized")
                
                df['primary_category'] = results
            
            elif method == "Configurable Keyword Matching":
                if not core_categories and not additional_categories:
                    st.error("Please enter at least one category and its variants")
                    return
                    
                df['search_text'] = df[text_columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
                df['primary_category'] = None
                df['match_score'] = 0
                df['matched_keywords'] = ""
                
                for core in [c.strip() for c in core_categories if c.strip()]:
                    for add in [c.strip() for c in additional_categories if c.strip()]:
                        # This is simplified - you'd need to implement your actual fuzzy matching logic here
                        df['primary_category'] = core + " + " + add
                
                matched = df[df['match_score'] >= min_score].copy()
                unmatched = df[df['match_score'] < min_score].copy()
                
            else:  # BERT
                texts = df[text_columns].apply(
                    lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist()
                
                # Load BERT model
                @st.cache_resource
                def load_bert_model():
                    return SentenceTransformer('all-MiniLM-L6-v2')
                
                model = load_bert_model()
                embeddings = model.encode(texts)
                
                if clustering_mode == "Keyword-Guided":
                    # Use keywords as initial cluster centers
                    seed_embeddings = model.encode(seed_keywords)
                    # (Add custom clustering logic here)
                    pass
                else:
                    # Standard clustering
                    clustering = AgglomerativeClustering(
                        n_clusters=num_clusters,
                        affinity='cosine',
                        linkage='average'
                    ).fit(embeddings)
                    df['primary_category'] = [f"Cluster {x+1}" for x in clustering.labels_]
            
            # POST-PROCESSING FOR ALL METHODS
            matched = df[df['primary_category'] != "Uncategorized"]
            unmatched = df[df['primary_category'] == "Uncategorized"]
            
            # STORE RESULTS
            st.session_state.processed_data = matched
            st.session_state.unmatched_data = unmatched
            st.session_state.categorization_results = {
                'matched': matched,
                'unmatched': unmatched,
                'show_results': True,
                'method': method
            }
            
            st.session_state.summaries.append({
                'stage': 3,
                'action': 'Primary categorization',
                'method': method,
                'matched_count': len(matched),
                'unmatched_count': len(unmatched),
                'categories_created': len(matched['primary_category'].unique()) if len(matched) > 0 else 0
            })
            
            # EXPORT TO EXCEL
            if export_excel:
                with pd.ExcelWriter("categorized_papers.xlsx") as writer:
                    for category in matched['primary_category'].unique():
                        sanitized = sanitize_sheet_name(str(category))
                        matched[matched['primary_category'] == category].to_excel(
                            writer, sheet_name=sanitized, index=False)
                    
                    if not unmatched.empty:
                        unmatched.to_excel(writer, sheet_name="Uncategorized", index=False)
                    
                    # Add summary sheet
                    summary = pd.DataFrame({
                        'Category': matched['primary_category'].value_counts().index,
                        'Count': matched['primary_category'].value_counts().values
                    })
                    summary.to_excel(writer, sheet_name="Summary", index=False)
                
                st.success("Excel export completed!")

            st.success("Categorization complete!")

    if st.session_state.categorization_results.get('show_results', False):
        matched = st.session_state.categorization_results['matched']
        unmatched = st.session_state.categorization_results['unmatched']
        method = st.session_state.categorization_results.get('method', 'Standalone Exact-Match')
        
        st.subheader("📊 Results Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Categorized Papers", len(matched))
        with col2:
            st.metric("Uncategorized Papers", len(unmatched))
        
        st.write(f"Method used: {method}")
        
        if show_details:
            if len(matched) > 0:
                st.subheader("Category Distribution")
                fig, ax = plt.subplots()
                matched['primary_category'].value_counts().plot(kind='bar', ax=ax)
                st.pyplot(fig)
                
                st.subheader("Sample Categorized Papers")
                if method == "Standalone Exact-Match":
                    st.dataframe(matched[['primary_category'] + text_columns].head())
                else:
                    st.dataframe(matched[['primary_category']].head())
            
            if len(unmatched) > 0:
                st.subheader("Unmatched Papers")
                st.dataframe(unmatched.head())
    
    if st.button("Continue to Stage 4", key="stage3_continue") and 'processed_data' in st.session_state:
        st.session_state.current_stage = 4
        st.rerun()
        
#Stage 4: Secondary Categorization
def stage_secondary_categorization():
    st.header("📚 Stage 4: Secondary Categorization")
    
    # Initialize session state
    if 'secondary_categorization_results' not in st.session_state:
        st.session_state.secondary_categorization_results = {
            'matched': pd.DataFrame(),
            'unmatched': pd.DataFrame(),
            'show_results': False
        }

    with st.expander("📝 Instructions", expanded=True):
        st.markdown("""
        **Purpose**: Create additional categorization dimensions for your papers.
        
        **How to use**:
        1. Enter your category names (one per line)
        2. For each category, click "Add Variants" to specify keywords
        3. Click "Categorize Papers" to process your data
        """)

    if st.session_state.processed_data is None:
        st.warning("Please complete Stage 3 first")
        return

    df = st.session_state.processed_data.copy()

    # Initialize category dictionary in session state
    if 'category_dict' not in st.session_state:
        st.session_state.category_dict = {}

    with st.sidebar:
        st.subheader("Category Setup")
        
        # Category name input
        category_names = st.text_area(
            "Enter category names (one per line):",
            "Methodology\nPopulation\nSetting",
            height=100,
            key="stage4_category_names"
        ).split('\n')
        
        # Variant input for each category
        st.subheader("Add Variants")
        selected_category = st.selectbox(
            "Select category to add variants:",
            [name.strip() for name in category_names if name.strip()],
            key="variant_category_select"
        )
        
        variants = st.text_area(
            f"Enter variants for '{selected_category}' (comma separated):",
            "qualitative, quantitative, mixed methods" if selected_category == "Methodology" else
            "children, adults, elderly" if selected_category == "Population" else
            "clinical, community, laboratory",
            key=f"variants_for_{selected_category}"
        )
        
        if st.button("Save Variants", key="save_variants"):
            if selected_category and variants:
                variant_list = [v.strip() for v in variants.split(',') if v.strip()]
                if variant_list:
                    st.session_state.category_dict[selected_category] = variant_list
                    st.success(f"Saved {len(variant_list)} variants for {selected_category}")
                else:
                    st.error("Please enter at least one variant")
            else:
                st.error("Please select a category and enter variants")

    # Show current categories and variants
    with st.expander("Current Categories & Variants", expanded=True):
        if st.session_state.category_dict:
            for category, variants in st.session_state.category_dict.items():
                st.write(f"**{category}**: {', '.join(variants)}")
        else:
            st.info("No categories/variants saved yet")

    if st.button("Categorize Papers", key="stage4_categorize"):
        if not st.session_state.category_dict:
            st.error("Please create at least one category with variants first")
            return
            
        with st.spinner("Categorizing papers..."):
            # Get text columns and combine
            text_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
            if not text_cols:
                st.error("No text columns found in the data")
                return
                
            df['search_text'] = df[text_cols].apply(
                lambda x: ' '.join(x.dropna().astype(str)), axis=1).str.lower()
            
            # Initialize results
            df['secondary_category'] = None
            df['secondary_keywords'] = ""
            
            # Process each category
            for category, terms in st.session_state.category_dict.items():
                pattern = '|'.join([re.escape(term.lower()) for term in terms])
                matches = df['search_text'].str.contains(pattern, na=False, regex=True)
                
                # Update category and keywords for matches
                df.loc[matches, 'secondary_category'] = category
                matched_terms = df.loc[matches, 'search_text'].str.findall(pattern).apply(
                    lambda x: ', '.join(set(x)))
                df.loc[matches, 'secondary_keywords'] = matched_terms

            matched = df[df['secondary_category'].notna()]
            unmatched = df[df['secondary_category'].isna()]

            st.session_state.processed_data = matched
            st.session_state.unmatched_data = unmatched
            st.session_state.secondary_categorization_results = {
                'matched': matched,
                'unmatched': unmatched,
                'show_results': True
            }
            
            st.session_state.summaries.append({
                'stage': 4,
                'action': 'Secondary categorization',
                'matched_count': len(matched),
                'unmatched_count': len(unmatched),
                'categories_created': len(st.session_state.category_dict)
            })

            st.success(f"Created {len(st.session_state.category_dict)} secondary categories!")

    # Display results
    if st.session_state.secondary_categorization_results.get('show_results', False):
        matched = st.session_state.secondary_categorization_results['matched']
        
        if not matched.empty:
            st.subheader("📊 Results Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Categorized Papers", len(matched))
            with col2:
                unmatched = st.session_state.secondary_categorization_results['unmatched']
                st.metric("Uncategorized Papers", len(unmatched) if not unmatched.empty else 0)
            
            if 'secondary_category' in matched.columns:
                unique_categories = matched['secondary_category'].unique()
                if len(unique_categories) > 0:
                    st.write(f"**Categories created:** {', '.join(unique_categories)}")
                    
                    st.subheader("Category Distribution")
                    fig, ax = plt.subplots()
                    matched['secondary_category'].value_counts().plot(kind='bar', ax=ax)
                    st.pyplot(fig)
                    
                    st.subheader("Sample Categorized Papers")
                    cols_to_show = ['secondary_category']
                    if 'secondary_keywords' in matched.columns:
                        cols_to_show.append('secondary_keywords')
                    if 'primary_category' in matched.columns:
                        cols_to_show.insert(0, 'primary_category')
                    if 'Title' in matched.columns:
                        cols_to_show.append('Title')
                    
                    st.dataframe(matched[cols_to_show].head(10))
                else:
                    st.warning("No categories were assigned to papers")
            else:
                st.warning("No category information found in results")
        else:
            st.warning("No papers were categorized")

    add_save_option(4)
    
    if st.button("Continue to Stage 5", key="stage4_continue"):
        st.session_state.current_stage = 5
        st.rerun()

# Stage 5: Final Analysis
def stage_final_analysis():
    st.header("📊 Stage 5: Final Analysis")

    with st.expander("📝 Instructions", expanded=True):
        st.markdown("""
        **Purpose**: Perform detailed analysis on your categorized papers.
        
        **How to use**:
        - **Manual Categorization**:
          1. Define analysis dimensions (e.g., Population, Methodology)
          2. For each dimension, define categories and keywords
          3. The system will classify papers based on your definitions
        
        - **Machine Learning Classification**:
          1. Choose between pre-trained models, training new models, or transformer models
          2. For new models, upload training data with text and labels
          3. For transformer models, define possible labels
          4. The system will classify papers automatically
        
        **Tips**:
        - Manual categorization works well when you know exactly what you're looking for
        - ML classification works well when you have examples to train on
        - Transformer models require no training but need clear label definitions
        """)

    if st.session_state.processed_data is None:
        st.warning("Please complete Stage 4 first")
        return

    df = st.session_state.processed_data.copy()
    analysis_mode = st.radio("Select analysis mode:", 
                           ["Manual categorization", "Machine learning classification"],
                           key="stage5_analysis_mode")

    if analysis_mode == "Manual categorization":
        with st.expander("Manual Analysis Settings", expanded=True):
            num_dimensions = st.number_input("Number of dimensions to analyze:", 1, 3, 2, key="stage5_num_dimensions")
            dimensions = {}
            for i in range(num_dimensions):
                dim_name = st.text_input(f"Dimension {i+1} name:", 
                                       value="",
                                       key=f"stage5_dim_{i}",
                                       help="Examples: 'Population', 'Methodology', 'Intervention'")
                if dim_name:
                    num_categories = st.number_input(f"Number of categories for {dim_name}:", 
                                                    2, 10, 3,
                                                    key=f"stage5_dim_{i}_num_categories")
                    categories = {}
                    for j in range(num_categories):
                        cat_name = st.text_input(f"Category {j+1} for {dim_name}:", 
                                               "",
                                               key=f"stage5_dim_{i}_cat_{j}",
                                               help="Examples: 'Children', 'Adults', 'Elderly'")
                        if cat_name:
                            keywords = st.text_area(f"Keywords for {cat_name}:", 
                                                  "",
                                                  key=f"stage5_dim_{i}_cat_{j}_keywords",
                                                  help="Comma-separated terms that identify this category").split(',')
                            categories[cat_name] = [kw.strip().lower() for kw in keywords if kw.strip()]
                    dimensions[dim_name] = categories

        if st.button("Run Manual Analysis", key="stage5_run_manual"):
            with st.spinner("Analyzing papers..."):
                if not dimensions:
                    st.error("Please define at least one dimension with categories")
                    return
                    
                text_cols = [col for col in df.columns if df[col].dtype == 'object']
                df['analysis_text'] = df[text_cols].apply(
                    lambda x: ' '.join(x.dropna().astype(str)), axis=1).str.lower()
                
                for dim, cats in dimensions.items():
                    df[dim] = "Unknown"
                    for cat, terms in cats.items():
                        if not terms:
                            continue
                        pattern = '|'.join([re.escape(term) for term in terms])
                        mask = df['analysis_text'].str.contains(pattern, na=False)
                        df.loc[mask, dim] = cat
                
                st.session_state.processed_data = df
                st.session_state.analysis_results = dimensions
                st.session_state.summaries.append({
                    'stage': 5,
                    'action': 'Manual analysis',
                    'dimensions': list(dimensions.keys())
                })
                st.success("Analysis completed!")

        if st.session_state.analysis_results:
            for dim in st.session_state.analysis_results:
                if dim in df.columns:
                    st.write(f"**{dim} Distribution**")
                    fig, ax = plt.subplots()
                    df[dim].value_counts().plot(kind='bar', ax=ax)
                    st.pyplot(fig)
                    
                    st.write(f"Sample papers for {dim}:")
                    # Safely check for columns before displaying
                    display_cols = [dim]
                    if 'primary_category' in df.columns:
                        display_cols.append('primary_category')
                    if 'secondary_category' in df.columns:
                        display_cols.append('secondary_category')
                    st.dataframe(df[display_cols].head())

    else:  # Machine learning classification
        with st.expander("ML Classification Settings", expanded=True):
            ml_option = st.radio(
                "ML Option:",
                ["Use pre-trained model", "Train new model", "Use transformer model"],
                key="stage5_ml_option"
            )

            if ml_option == "Use transformer model":
                model_type = st.selectbox(
                    "Select transformer model:",
                    ["SciBART", "BERT", "DistilBERT"],
                    key="stage5_transformer_type"
                )
                target_dimension = st.text_input("Enter dimension name:", 
                                               "",
                                               key="stage5_transformer_dimension",
                                               help="What you're classifying (e.g., 'Population')")
                candidate_labels = st.text_area(
                    "Enter possible labels (comma-separated):", 
                    "",
                    key="stage5_transformer_labels",
                    help="Possible values for this dimension (e.g., 'Children,Adults,Elderly')"
                ).split(',')
                candidate_labels = [label.strip() for label in candidate_labels if label.strip()]
                
            elif ml_option == "Use pre-trained model":
                target_dimension = st.text_input(
                    "Enter dimension name:", 
                    "",
                    key="stage5_pretrained_dimension",
                    help="What you're classifying (e.g., 'Population')"
                )
                
            elif ml_option == "Train new model":
                training_file = st.file_uploader(
                    "Upload training data (Excel/CSV with text and labels):", 
                    type=['csv', 'xlsx'],
                    key="stage5_training_file"
                )

                if training_file:
                    if training_file.name.endswith(".csv"):
                        train_df = pd.read_csv(training_file)
                    else:
                        train_df = pd.read_excel(training_file)

                    if train_df is not None:
                        text_col = st.selectbox("Select text column:", 
                                              train_df.columns,
                                              key="stage5_text_col")
                        label_col = st.selectbox("Select label column:", 
                                               train_df.columns,
                                               key="stage5_label_col")
                        target_dimension = st.text_input("Name for this dimension:", 
                                                       value=label_col,
                                                       key="stage5_new_dimension")

                        st.subheader("Model Parameters")
                        test_size = st.slider("Test set size (%):", 
                                            10, 40, 20,
                                            key="stage5_test_size")
                        n_estimators = st.slider("Number of trees:", 
                                                50, 500, 100,
                                                key="stage5_n_estimators")

        if st.button("Run ML Classification", key="stage5_run_ml"):
            with st.spinner("Running classification..."):
                try:
                    text_cols = [col for col in df.columns if df[col].dtype == 'object']
                    df['ml_text'] = df[text_cols].apply(
                        lambda x: ' '.join(x.dropna().astype(str)), axis=1)
                    df = df.dropna(subset=['ml_text'])

                    if ml_option == "Use pre-trained model":
                        if not target_dimension:
                            st.error("Please enter a dimension name")
                            return
                            
                        try:
                            model = joblib.load(f"{target_dimension.lower()}_classifier.pkl")
                            vectorizer = joblib.load(f"{target_dimension.lower()}_vectorizer.pkl")
                            X = vectorizer.transform(df['ml_text'])
                            df[target_dimension] = model.predict(X)
                        except FileNotFoundError:
                            st.error(f"No pre-trained model found for '{target_dimension}'")
                            return

                    elif ml_option == "Use transformer model":
                        if not target_dimension or not candidate_labels:
                            st.error("Please enter a dimension name and candidate labels")
                            return
                            
                        model_map = {
                            "SciBART": "allenai/scibart",
                            "BERT": "bert-base-uncased",
                            "DistilBERT": "distilbert-base-uncased"
                        }
                        
                        try:
                            classifier = transformers_pipeline(
                                "zero-shot-classification",
                                model=model_map[model_type],
                                device=0 if torch.cuda.is_available() else -1
                            )
                            
                            df[target_dimension] = df['ml_text'].apply(
                                lambda text: classifier(
                                    text, 
                                    candidate_labels,
                                    multi_label=False
                                )['labels'][0] if text else "Unknown"
                            )
                        except Exception as e:
                            st.error(f"Failed to load {model_type} model: {str(e)}")
                            st.info("Falling back to BERT model")
                            classifier = transformers_pipeline(
                                "zero-shot-classification",
                                model="bert-base-uncased"
                            )
                            df[target_dimension] = df['ml_text'].apply(
                                lambda text: classifier(text, candidate_labels)['labels'][0] 
                                if text else "Unknown"
                            )

                    elif ml_option == "Train new model":
                        if training_file is None or not text_col or not label_col:
                            st.error("Please upload training data and select text/label columns")
                            return
                            
                        X_train, X_test, y_train, y_test = train_test_split(
                            train_df[text_col], train_df[label_col], 
                            test_size=test_size/100, random_state=42
                        )
                        pipeline_model = Pipeline([
                            ('tfidf', TfidfVectorizer()),
                            ('clf', RandomForestClassifier(n_estimators=n_estimators))
                        ])
                        pipeline_model.fit(X_train, y_train)
                        df[target_dimension] = pipeline_model.predict(df['ml_text'])

                    st.session_state.processed_data = df
                    st.session_state.summaries.append({
                        'stage': 5,
                        'action': 'ML classification',
                        'dimension': target_dimension,
                        'method': ml_option
                    })
                    st.success("Classification completed!")
                    
                    if target_dimension in df.columns:
                        st.subheader(f"{target_dimension} Distribution")
                        fig, ax = plt.subplots()
                        df[target_dimension].value_counts().plot(kind='bar', ax=ax)
                        st.pyplot(fig)
                        
                        st.subheader("Sample Results")
                        display_cols = [target_dimension, 'ml_text']
                        if 'primary_category' in df.columns:
                            display_cols.append('primary_category')
                        if 'secondary_category' in df.columns:
                            display_cols.append('secondary_category')
                        st.dataframe(df[display_cols].head())

                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
    
    add_save_option(5)
    
    if st.button("Continue to Export", key="stage5_continue") and 'processed_data' in st.session_state:
        st.session_state.current_stage = 6
        st.rerun()

# Stage 6: Export Results
def stage_export_results():
    st.header("💾 Stage 6: Export Results")
    
    with st.expander("📝 Instructions", expanded=True):
        st.markdown("""
        **Purpose**: Export your processed and analyzed research papers.
        
        **How to use**:
        1. Review the pipeline summary
        2. Select which columns to include in the export
        3. Choose your preferred export format (Excel, CSV, or both)
        4. Enter a base name for your exported files
        5. Click "Generate Export Files"
        6. Download your files when ready
        
        **Tips**:
        - Excel format preserves multiple sheets (processed data, unmatched data, summary)
        - CSV format is good for simple single-file exports
        - You can restart the pipeline to process new data
        """)

    if st.session_state.processed_data is None:
        st.warning("No data to export. Please complete previous stages.")
        return
    
    st.subheader("Pipeline Summary")
    summary_df = pd.DataFrame(st.session_state.summaries)
    st.dataframe(summary_df)
    
    df = st.session_state.processed_data.copy()
    
    export_format = st.radio("Select export format:", 
                           ["Excel (multiple sheets)", "CSV (single file)", "Both"],
                           key="stage6_export_format")
    export_name = st.text_input("Base name for exported files:", 
                              "research_papers_analysis",
                              key="stage6_export_name")

    # Get available columns safely
    all_columns = df.columns.tolist()
    default_columns = []
    
    # Build default columns from available data
    for col in ['Title', 'Abstract', 'primary_category', 'secondary_category']:
        if col in all_columns:
            default_columns.append(col)
    
    if not default_columns:
        default_columns = all_columns[:min(5, len(all_columns))]
    
    selected_columns = st.multiselect(
        "Select columns to include in export:", 
        all_columns, 
        default=default_columns,
        key="stage6_selected_columns"
    )
    
    if st.button("Generate Export Files", key="stage6_generate_export"):
        with st.spinner("Preparing export..."):
            try:
                # Filter data based on selected columns
                export_df = df[selected_columns] if selected_columns else df
                
                # Handle unmatched data if it exists
                unmatched_export = None
                if 'unmatched_data' in st.session_state and not st.session_state.unmatched_data.empty:
                    # Only include columns that exist in unmatched data
                    unmatched_cols = [col for col in selected_columns 
                                    if col in st.session_state.unmatched_data.columns]
                    if unmatched_cols:
                        unmatched_export = st.session_state.unmatched_data[unmatched_cols]
                    else:
                        unmatched_export = st.session_state.unmatched_data
                
                # Excel export
                if export_format in ["Excel (multiple sheets)", "Both"]:
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        export_df.to_excel(writer, sheet_name='Processed_Papers', index=False)
                        
                        if unmatched_export is not None:
                            unmatched_export.to_excel(writer, sheet_name='Unmatched_Papers', index=False)
                        
                        summary_df.to_excel(writer, sheet_name='Pipeline_Summary', index=False)
                        
                        # Add visualization
                        workbook = writer.book
                        worksheet = writer.sheets['Pipeline_Summary']
                        chart = workbook.add_chart({'type': 'column'})
                        chart.add_series({
                            'name': 'Records Processed',
                            'categories': '=Pipeline_Summary!$B$2:$B$' + str(len(summary_df)+1),
                            'values': '=Pipeline_Summary!$D$2:$D$' + str(len(summary_df)+1),
                        })
                        worksheet.insert_chart('D2', chart)
                    
                    excel_buffer.seek(0)
                    excel_bytes = excel_buffer.getvalue()
                
                # CSV export
                if export_format in ["CSV (single file)", "Both"]:
                    csv_buffer = BytesIO()
                    export_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    csv_bytes = csv_buffer.getvalue()
                
                # Create download buttons
                if export_format == "Both":
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                        zip_file.writestr(f"{export_name}.xlsx", excel_bytes)
                        zip_file.writestr(f"{export_name}.csv", csv_bytes)
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="📦 Download ZIP File (Excel + CSV)",
                        data=zip_buffer,
                        file_name=f"{export_name}.zip",
                        mime="application/zip",
                        key="stage6_zip_download"
                    )
                elif export_format == "Excel (multiple sheets)":
                    st.download_button(
                        label="📥 Download Excel File",
                        data=excel_bytes,
                        file_name=f"{export_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="stage6_excel_download"
                    )
                else:  # CSV
                    st.download_button(
                        label="📥 Download CSV File",
                        data=csv_bytes,
                        file_name=f"{export_name}.csv",
                        mime="text/csv",
                        key="stage6_csv_download"
                    )
                
                st.success("Export files ready!")
            
            except Exception as e:
                st.error(f"Error during export: {str(e)}")

    if st.button("🔄 Restart Pipeline", key="stage1_restart"):
        reset_pipeline()


def pipeline_controller():
    st.sidebar.header("📊 Pipeline Progress")
    stages = [
        "1. Data Collection & EDA",
        "2. Duplicate Removal",
        "3. Primary Categorization",
        "4. Secondary Categorization",
        "5. Final Analysis",
        "6. Export Results"
    ]
    for i, stage in enumerate(stages, 1):
        if i < st.session_state.current_stage:
            st.sidebar.success(f"✔️ {stage}")
        elif i == st.session_state.current_stage:
            st.sidebar.info(f"➡️ {stage}")
        else:
            st.sidebar.write(stage)

    if st.sidebar.button("🔄 Restart Pipeline", key="sidebar_restart_button"):
        reset_pipeline()

    # Progress bar
    progress = (st.session_state.current_stage - 1) / (len(stages) - 1)
    st.progress(progress)

    # Stage routing
    if st.session_state.current_stage == 1:
        stage_data_eda()
    elif st.session_state.current_stage == 2:
        stage_duplicate_removal()
    elif st.session_state.current_stage == 3:
        stage_primary_categorization()
    elif st.session_state.current_stage == 4:
        stage_secondary_categorization()
    elif st.session_state.current_stage == 5:
        stage_final_analysis()
    elif st.session_state.current_stage == 6:
        stage_export_results()

def main():
    st.set_page_config(
        page_title="A fully Automated Systematic Review Application",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 A fully Automated Systematic Review Application")
    st.write("Developed by **Kanwar Hamza Shuja**")
    st.write("Email: kanwarhamzashuja@gmail.com; kanwarhamza.shuja@unito.it")
    st.write("https://github.com/KanwarHamza")
    st.write("www.linkedin.com/in/psy-kanwar-hamza-shuja1")
    st.write("7/05/2025")

    st.markdown("""
    A complete workflow for processing and analyzing research papers:
    1. **Data Collection & EDA** → 2. **Duplicate Removal** → 
    3. **Primary Categorization** → 4. **Secondary Categorization** → 
    5. **Final Analysis** → 6. **Export Results**
    
    **Getting Started**:
    - Begin by uploading your dataset or downloading from scientific APIs
    - Follow the instructions at each stage
    - Use the EDA report to understand your data before processing
    - Save your progress at any stage using the sidebar
    """)

    pipeline_controller()

if __name__ == "__main__":
    main()
