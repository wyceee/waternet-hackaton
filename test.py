from openai import OpenAI
import os
from typing import Optional, Dict, Any
import pandas as pd
import json


with open("openai.key") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)
print(client.models.list())  # should return available models if the key works

MODEL = "gpt-4o-mini"  # change to a model you have access to

# -----------------------------
# 1) Load CSV
# -----------------------------
def load_water_data(csv_path: str) -> pd.DataFrame:
    """Read the water quality CSV and do light parsing."""
    df = pd.read_csv(csv_path, sep=';')  # Your data uses semicolon separator
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    return df

# -----------------------------
# 2) (Optional) Filter with pandas
# -----------------------------
def filter_water_data(
    df: pd.DataFrame,
    species: Optional[str] = None,
    location: Optional[str] = None,
    year: Optional[int] = None,
    parameter: Optional[str] = None,
) -> pd.DataFrame:
    """Filter water quality data by various criteria."""
    out = df.copy()

    if species and "biotaxonnaam" in out.columns:
        out = out[out["biotaxonnaam"].str.contains(species, case=False, na=False)]

    if location and "locatie omschrijving" in out.columns:
        out = out[out["locatie omschrijving"].str.contains(location, case=False, na=False)]

    if parameter and "fewsparameternaam" in out.columns:
        out = out[out["fewsparameternaam"].str.contains(parameter, case=False, na=False)]

    if year and "datum" in out.columns:
        out = out[out["datum"].dt.year == year]

    return out

# -----------------------------
# 2-alternative) Have the LLM suggest filters
# -----------------------------
def llm_suggest_filters(question: str, df_columns: list[str]) -> Dict[str, Any]:
  """
  Ask the LLM to extract filters from a free-text question about water quality data.
  Returns a dict with keys: species, location, year, parameter (or None if unknown).
  """
  # Only ask for fields that exist in the CSV
  want_species = "biotaxonnaam" in df_columns
  want_location = "locatie omschrijving" in df_columns
  want_year = "datum" in df_columns
  want_parameter = "fewsparameternaam" in df_columns

  system = (
      "Extract filters from the user's question for a water quality monitoring CSV.\n"
      "Only output a single JSON object with keys: species, location, year, parameter.\n"
      "Use null if unknown or not specified. Year must be an integer (e.g., 1998) or null.\n"
      "Species refers to biological species names (biotaxonnaam).\n"
      "Location refers to monitoring site descriptions.\n"
      "Parameter refers to measurement types."
  )

  msg_user = (
      "Question:\n"
      f"{question}\n\n"
      "Return JSON ONLY, no extra text. Example:\n"
      '{"species":"Anisus vortex","location":"volkstuinen","year":1998,"parameter":"Macro"}'
  )

  resp = client.chat.completions.create(
      model=MODEL,
      temperature=0.1,
      messages=[
          {"role": "system", "content": system},
          {"role": "user", "content": msg_user},
      ]
  )

  text = resp.choices[0].message.content.strip()
  try:
      data = json.loads(text)
  except Exception:
      # Safe fallback if model didn't return proper JSON
      data = {"species": None, "location": None, "year": None, "parameter": None}

  # Force to our shape/types
  out = {
      "species": (data.get("species") if isinstance(data.get("species"), str) and data.get("species").strip() else None),
      "location": (data.get("location") if isinstance(data.get("location"), str) and data.get("location").strip() else None),
      "parameter": (data.get("parameter") if isinstance(data.get("parameter"), str) and data.get("parameter").strip() else None),
      "year": (int(data.get("year")) if isinstance(data.get("year"), (int, str)) and str(data.get("year")).isdigit() else None),
  }
  # Respect missing columns
  if not want_species: out["species"] = None
  if not want_location: out["location"] = None
  if not want_parameter: out["parameter"] = None
  if not want_year: out["year"] = None
  return out

def filter_water_data_auto(df, question: str):
    """
    Uses the LLM to infer filters from the question, then applies filter_water_data.
    """
    inferred = llm_suggest_filters(question, df.columns.tolist())
    return filter_water_data(
        df,
        species=inferred["species"],
        location=inferred["location"],
        year=inferred["year"],
        parameter=inferred["parameter"],
    )

# -----------------------------
# 3) Build a tiny prompt
# -----------------------------
def build_prompt(user_question: str, rows: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Keep things simple: include up to N rows as CSV text.
    For large datasets, you'd filter first, then include just what you need.
    """
    snippet = rows.head(max_rows).to_csv(index=False, sep=';')
    instructions = (
        "You are a helpful water quality data analyst. Use ONLY the rows below to answer. "
        "If the answer is not in the rows, say you don't know.\n"
        "The data contains information about water quality measurements including species counts, "
        "locations, dates, and measurement values.\n\n"
        "ROWS (CSV):\n"
    )
    return f"{instructions}{snippet}\nQUESTION: {user_question}"

# -----------------------------
# 4) Ask the model
# -----------------------------
def ask_llm(prompt: str) -> str:
    """One simple chat completion call."""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "Be concise and factual."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# 5) Put it together
# -----------------------------
def ask_about_water_data(
    csv_path: str,
    question: str,
    species: Optional[str] = None,
    location: Optional[str] = None,
    year: Optional[int] = None,
    parameter: Optional[str] = None,
    max_rows: int = 20,
) -> str:
    """
    One convenience function:
    - load CSV,
    - (optionally) filter,
    - build prompt,
    - ask model.
    """
    print('-'*50)
    print(f'Answering question: {question}')
    df = load_water_data(csv_path)
    print(f'Loaded {len(df)} rows from CSV')
    
    # Apply manual filters if provided
    df_filtered = filter_water_data(df, species=species, location=location, year=year, parameter=parameter)
    print(f'Manual filtering: {len(df_filtered)} rows')
    
    # Apply automatic LLM-based filtering
    df_auto = filter_water_data_auto(df, question)
    print(f'Auto filtering: {len(df_auto)} rows')
    
    # Use the auto-filtered data
    prompt = build_prompt(question, df_auto, max_rows=max_rows)
    print('-'*50)
    return ask_llm(prompt)

# -----------------------------
# 6) Example usage
# -----------------------------
if __name__ == "__main__":
    # Test the system
    csv_file = "HB100sampled.csv"
    
    # Example questions you can ask
    questions = [
        "What species were found at volkstuinen Amstelglorie?",
        "How many different species were observed in 1998?",
        "What was the count of Anisus vortex found?",
        "Which locations had the highest species diversity?",
        "What macro-invertebrate species were most common?"
    ]
    
    print("Testing water quality data analysis...")
    
    for question in questions[:2]:  # Test first 2 questions
        try:
            answer = ask_about_water_data(csv_file, question)
            print(f"\nQ: {question}")
            print(f"A: {answer}")
            print("="*50)
        except Exception as e:
            print(f"Error with question '{question}': {e}")

