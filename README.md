[README_TFM.md](https://github.com/user-attachments/files/29145116/README_TFM.md)
# Topic Evolution and Discussion Structure Analysis in Online Communities Using Large Language Models

Master's Thesis project by **J. Narciso de la Llama**.

This repository contains the complete empirical pipeline used to collect, preprocess, classify, validate and analyse political communication on X/Twitter. The study focuses on topic evolution, stance detection, agenda similarity, temporal dynamics and thematic graph construction using Large Language Models and traditional NLP baselines.

## Project overview

The corpus contains tweets published by four Spanish political actors:

- Pedro Sánchez, `sanchezcastejon`
- Alberto Núñez Feijóo, `NunezFeijoo`
- Santiago Abascal, `Santi_ABASCAL`
- Yolanda Díaz, `Yolanda_Diaz_`

The observation period extends from **23 April 2022 to 23 April 2026**. After preprocessing and filtering, the final classified corpus contains **11,866 unique tweets**.

Each tweet was assigned:

- one topic label,
- one stance label,
- a confidence score,
- a short justification.

The classification model was **Gemini `gemini-3.1-flash-lite-preview`**.

## Classification schema

### Topics

1. Economy and Employment
2. Welfare, Housing and Social Policy
3. National Politics and Governance
4. International Affairs
5. Immigration and Security
6. Rights and Equality
7. Other

### Stances

1. In favor
2. Against
3. Neutral
4. Unclear

## Repository contents

### Data collection

#### `ChromeUser2.py`

Opens Chrome with a persistent local profile so that the user can log in manually to X/Twitter. The saved session is reused by the scraper.

Before running it, update:

```python
CHROME_PROFILE_PATH = r"path\to\your\chrome_selenium_profile"
```

Run with:

```bash
python ChromeUser2.py
```

#### `scraper_rangos.py`

Collects tweets through X/Twitter search pages using Selenium. The script:

- searches each account separately,
- divides the observation period into 45-day windows,
- scrolls through dynamically loaded results,
- extracts tweet IDs, text, dates and URLs,
- detects original tweets, replies and reposts,
- removes repeated URLs during collection,
- saves progress incrementally as JSON.

Local paths and actor/date settings are defined in `ScraperConfig`.

Run with:

```bash
python scraper_rangos.py
```

### Preprocessing

#### `Preprocessing_final.py`

Transforms the raw JSON files into an LLM-ready CSV. It performs:

- text cleaning,
- URL replacement with `[URL]`,
- user mention replacement with `[USER]`,
- reply and repost context construction,
- date and time feature creation,
- word and character counts,
- duplicate removal using normalized-text hashes,
- creation of the `usable_for_llm` flag.

Input folder:

```text
data_raw/
```

Output:

```text
complete_tweets_clean_llm_ready.csv
```


#### `complete_tweets_clean_llm_ready.csv`

Preprocessed corpus containing **12,261 rows** before the final LLM usability filter. It includes tweet metadata, normalized text, temporal variables and input-quality indicators.

### Prompt evaluation and LLM classification

#### `ComparePrompts.py`

Runs an A/B test between two prompt versions on a stratified sample of 500 tweets. It compares:

- topic agreement,
- stance agreement,
- Cohen's kappa,
- confidence distributions,
- changes in stance labels,
- frequency of the `Unclear` category.

Output:

```text
prompt_ab_test_results.csv
```

The output contains 1,000 rows because each of the 500 tweets is classified with both prompt versions.

#### `prompt_ab_test_results.csv`

Stores the complete output of the prompt comparison experiment, including prompt version, model name, topic, stance, confidence and justification.

#### `classify_tweets.py`

Main Gemini classification script. It:

- processes tweets in batches of 25,
- enforces a structured Pydantic output schema,
- controls requests per minute and per day,
- retries temporary API errors,
- saves results incrementally,
- resumes from checkpoints,
- returns one topic and one stance per tweet.

The API key must be stored in a local `.env` file:

```env
GEMINI_API_KEY2=your_api_key_here
```

Main input:

```text
complete_tweets_clean_llm_ready.csv
```

Checkpoint:

```text
classify_checkpoint.csv
```


### Final classified datasets

#### `classified_all_tweets_final_v2.csv`

Full classified corpus with 11,866 tweets and the following main fields:

- `row_id`
- `tweet_id`
- `tweet`
- `topic`
- `stance`
- `confidence`
- `short_justification`
- `date`
- `politician`

#### `classified_all_tweets_final_v2_clean.csv`

Clean analytical version of the classified corpus. It contains the final topic and stance labels together with cleaned temporal variables. This is the preferred input for the notebooks and dashboard.

## Analysis notebooks

### `EdaDataset.ipynb`

Performs final data checks and exploratory validation, including:

- missing or invalid labels,
- duplicate checks,
- date coverage,
- actor coverage,
- confidence inspection,
- full-period and common-period comparisons,
- basic temporal summaries.

### `corpus_description.ipynb`

Describes the final corpus and actor-level communication profiles. It includes:

- global topic and stance distributions,
- tweet volume and temporal coverage,
- monthly activity timeline,
- actor-topic matrices,
- Jensen-Shannon agenda divergence,
- actor-level stance distributions,
- positivity index,
- audit of the `Other` category.

### `build_gold_standard_sample.ipynb`

Creates the human-validation sample. It:

- draws a stratified sample,
- increases representation of rare categories,
- creates blind annotation files,
- adds Excel dropdowns for topic and stance labels,
- stores researcher-only sampling metadata.

### `gold_standard_annotator_A.xlsx`

Blind annotation workbook for Annotator A. It contains 200 tweets and empty fields for:

- `annotator_topic`
- `annotator_stance`
- `annotator_notes`

The workbook also contains a label sheet used for data validation.

### `gold_standard_evaluation.ipynb`

Evaluates human and model agreement. It computes:

- accuracy,
- Cohen's kappa,
- precision,
- recall,
- F1 score,
- confusion matrices,
- disagreement patterns,
- inter-annotator agreement.

### `gold_standard_comparison_results.xlsx`

Contains the final comparison results for Annotator A, Annotator B and a summary sheet with the main evaluation metrics.

### `nlp_comparison.ipynb`

Compares Gemini with traditional and transformer-based NLP methods:

- Logistic Regression with TF-IDF,
- Support Vector Machine with TF-IDF,
- Multinomial Naive Bayes with TF-IDF,
- multilingual mDeBERTa zero-shot classification,
- Gemini zero-shot classification.

The notebook evaluates both topic and stance classification using accuracy and Cohen's kappa.

### `temporal_evolution.ipynb`

Studies topic evolution over time using monthly actor-topic series. It includes:

- topic trajectories by actor,
- stacked agenda composition,
- topic-by-topic actor comparisons,
- changepoint exploration,
- cross-correlation and lead-lag analysis,
- exploratory Granger tests,
- interpretation of major political events.

The causal analyses are exploratory and are not treated as evidence of causal agenda influence.

### `thematic_graphs.ipynb`

Builds interpretable graph-based representations:

- actor-topic bipartite network,
- actor agenda-similarity network,
- topic-topic correlation network,
- stance-polarization heatmap,
- temporal agenda-similarity snapshots.

## Interactive dashboard

### `dashboard.py`

Streamlit application for exploring the final classified corpus. It includes seven tabs:

1. Corpus Overview
2. Distributions
3. Actor Comparison
4. Temporal Evolution
5. Thematic Graphs
6. Tweet Explorer
7. About

Available filters include:

- actor,
- topic,
- stance,
- date range,
- confidence range,
- counts or percentages.

The dashboard loads the cleaned classified dataset from the current folder or from a `data/` subfolder.



## Recommended execution order

To reproduce the full pipeline:

1. `ChromeUser2.py`
2. `scraper_rangos.py`
3. `Preprocessing_final.py`
4. `ComparePrompts.py`
5. `classify_tweets.py`
6. `build_gold_standard_sample.ipynb`
7. Complete the human annotation workbooks
8. `gold_standard_evaluation.ipynb`
9. `EdaDataset.ipynb`
10. `corpus_description.ipynb`
11. `nlp_comparison.ipynb`
12. `temporal_evolution.ipynb`
13. `thematic_graphs.ipynb`
14. `dashboard.py`

## Installation

A typical environment can be created with:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the main dependencies:

```bash
pip install pandas numpy matplotlib plotly streamlit networkx scikit-learn openpyxl selenium python-dotenv pydantic google-genai statsmodels scipy ruptures transformers torch jupyter
```

Some notebooks may use additional visualization or utility packages depending on the local environment.

## Reproducibility notes

- API keys are not included.
- Store secrets in a local `.env` file and never commit them.
- X/Twitter scraping requires a valid logged-in browser profile.
- Selenium selectors may need adjustment if the X interface changes.
- Historical search results may not provide a complete archive of every published tweet.
- The Gemini model was a preview model and may later change or become unavailable.
- The final datasets are included so that the analysis can be reviewed without repeating scraping or API classification.
- Random seeds are fixed in sampling and model evaluation where applicable.
- Some notebook paths may need to be adapted to the local folder structure.

## Main findings

The final analysis identified several consistent patterns:

- National Politics and Governance was the most frequent topic.
- Feijóo and Abascal concentrated strongly on national politics.
- Yolanda Díaz gave greater attention to Economy and Employment.
- Pedro Sánchez showed the largest relative presence of International Affairs.
- Sánchez and Díaz had the most similar topic distributions, followed by Feijóo and Abascal.
- Government actors showed more supportive stance profiles, while Abascal showed the strongest oppositional profile.
- Gemini outperformed the evaluated traditional NLP baselines under the selected experimental conditions.
- Human disagreement remained relevant, especially for minority topics and the `Neutral` and `Unclear` stance categories.
- Temporal correlations and thematic graphs were interpreted descriptively rather than causally.

## Scope and limitations

This repository studies the public communication of four selected political actors. It does not represent the complete Spanish political system or the full public conversation on X/Twitter.

The classification scheme is single-label, so tweets containing multiple topics or stance targets are simplified to one dominant label. Model-generated confidence values are auxiliary metadata and should not be interpreted as calibrated probabilities.

## Author

**J. Narciso de la Llama**

Master's Degree in Big Data Analytics  
Universidad Carlos III de Madrid
