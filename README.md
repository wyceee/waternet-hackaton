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

## Command-line test

```bash
python test.py
```

## Streamlit UI

Run the interactive app:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

In the sidebar you can:
1. Enter / override your API key
2. Upload a custom semicolon-separated CSV (or use the bundled sample `HB100sampled.csv`)
3. Optionally specify manual filters
4. Ask natural language questions; the app auto-suggests filters via the model

Environment variables:
* `OPENAI_API_KEY` – API key (optional if using `openai.key` file)
* `OPENAI_MODEL` – override model name (default: `gpt-4o-mini`)

