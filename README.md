## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Provide an OpenAI API key either via env var or a file:

```bash
echo YOUR_OPENAI_KEY > openai.key
# or
export OPENAI_API_KEY=YOUR_OPENAI_KEY
```

## Streamlit UI

Run the interactive app:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

In the sidebar you can:
1. Enter / override your API key
2. Upload a custom semicolon-separated CSV (or use the bundled sample `HB100sampled2.csv`)
3. Optionally specify manual filters
4. Ask natural language questions; the app auto-suggests filters via the model

Environment variables:
* `OPENAI_API_KEY` – API key (optional if using `openai.key` file)
* `OPENAI_MODEL` – override model name (default: `gpt-5`)

Notes:
* CSVs are expected to be semicolon (`;`) separated.
* The model is prompted to infer filters (species/location/year/parameter) from your natural language question.
* Increase "Max rows in prompt" for broader context, at the cost of more tokens.

## How It Works

High‑level question → answer pipeline (when you press "Ask Model"):

1. Data acquisition
	* Either an uploaded CSV or one (or several concatenated) repository CSVs is loaded with `load_water_data`.
	* Dates in a `datum` column are parsed to `datetime` if present.
2. Optional manual filters
	* Substring (case‑insensitive) filters for species (`biotaxonnaam`), location (`locatie omschrijving`), parameter (`fewsparameternaam`) and an exact year match on the parsed date.
	* Implemented by `filter_water_data` (no regex beyond simple `.str.contains`, no multi‑value logic, no ranges).
3. LLM filter suggestion
	* `llm_suggest_filters` sends your natural language question and a strict system prompt to the chat model, asking for a JSON object with: `species`, `location`, `year`, `parameter`.
	* The JSON is parsed; empty strings become `None`; `year` is coerced to an `int` if all digits.
	* Any key whose corresponding column is missing is nulled out.
4. Automatic filtering
	* The suggested values are passed again to `filter_water_data` to narrow the DataFrame (`auto_df`).
	* If the suggestion is overly specific and matches few/no rows, the prompt will contain only those rows (possibly zero ⇒ the model is expected to reply it doesn’t know).
5. Prompt construction
	* `build_prompt` converts the (filtered) DataFrame head to CSV (semicolon‑separated) and prepends concise instructions telling the model to ONLY use those rows.
	* Only the first `max_rows` (slider) rows are included: there is currently no semantic ranking, sampling, or column pruning—just a truncation via `DataFrame.head(max_rows)`.
6. Model answer
	* `ask_llm` sends the assembled prompt with a concise system instruction.
	* The response text is shown directly.

### Limitations & Considerations
* Relevance Selection: No semantic similarity or keyword scoring—only positional `head(max_rows)` after filtering.
* Filter Accuracy: The model might propose values not present in the data; this can shrink the context inadvertently.
* Single Value Only: `llm_suggest_filters` extracts at most one value per field (no lists / OR logic).
* Year Handling: Only a single exact year integer; no ranges.
* Token Control: Increase `max_rows` for more context; consider adding future ranking/summarization to stay under token limits.
* Determinism: LLM extraction adds variability; a deterministic keyword fallback could be added if reproducibility is required.

### Possible Enhancements (Not Implemented Yet)
* Keyword / fuzzy matching fallback when LLM suggests filters yielding 0 rows.
* Ranking rows by keyword overlap with the question before truncation.
* Summarizing large groups of similar rows to save tokens.
* Allowing multiple species / locations (OR logic) or year ranges.
* Column pruning (include only relevant columns in the prompt).

