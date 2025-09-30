"""Core water quality data + LLM helper functions.

Minimal usage example:

    from water_core import init_openai, ask_about_water_data
    init_openai(api_key)
    answer = ask_about_water_data("HB100sampled.csv", "Your question")
    print(answer)
"""

from __future__ import annotations

from openai import OpenAI
import pandas as pd
from typing import Optional, Dict, Any, List
import json
import os

# The global client is lazily initialised via init_openai to avoid side effects
client: OpenAI | None = None
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # allow override via env


def init_openai(api_key: str) -> None:
    """Initialise the global OpenAI client once."""
    global client
    if client is None:
        client = OpenAI(api_key=api_key)


def load_water_data(csv_path: str) -> pd.DataFrame:
    """Read the water quality CSV and do light parsing (semicolon separated)."""
    df = pd.read_csv(csv_path, sep=';')
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    return df


def filter_water_data(
    df: pd.DataFrame,
    species: Optional[str] = None,
    location: Optional[str] = None,
    year: Optional[int] = None,
    parameter: Optional[str] = None,
) -> pd.DataFrame:
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


def _require_client() -> OpenAI:
    if client is None:
        raise RuntimeError("OpenAI client not initialised. Call init_openai(api_key) first.")
    return client


def llm_suggest_filters(question: str, df_columns: List[str]) -> Dict[str, Any]:
    _client = _require_client()
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
    resp = _client.chat.completions.create(
        model=MODEL,
        temperature=1.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": msg_user},
        ],
    )
    text = resp.choices[0].message.content.strip()
    try:
        data = json.loads(text)
    except Exception:
        data = {"species": None, "location": None, "year": None, "parameter": None}
    out = {
        "species": (data.get("species") if isinstance(data.get("species"), str) and data.get("species").strip() else None),
        "location": (data.get("location") if isinstance(data.get("location"), str) and data.get("location").strip() else None),
        "parameter": (data.get("parameter") if isinstance(data.get("parameter"), str) and data.get("parameter").strip() else None),
        "year": (int(data.get("year")) if isinstance(data.get("year"), (int, str)) and str(data.get("year")).isdigit() else None),
    }
    if not want_species: out["species"] = None
    if not want_location: out["location"] = None
    if not want_parameter: out["parameter"] = None
    if not want_year: out["year"] = None
    return out


def filter_water_data_auto(df: pd.DataFrame, question: str) -> pd.DataFrame:
    inferred = llm_suggest_filters(question, df.columns.tolist())
    return filter_water_data(
        df,
        species=inferred["species"],
        location=inferred["location"],
        year=inferred["year"],
        parameter=inferred["parameter"],
    )


def build_prompt(user_question: str, rows: pd.DataFrame, max_rows: int = 20) -> str:
    snippet = rows.head(max_rows).to_csv(index=False, sep=';')
    instructions = (
        "You are a helpful water quality data analyst. Use ONLY the rows below to answer. "
        "If the answer is not in the rows, say you don't know.\n"
        "The data contains information about water quality measurements including species counts, "
        "locations, dates, and measurement values.\n\n"
        "ROWS (CSV):\n"
    )
    return f"{instructions}{snippet}\nQUESTION: {user_question}"


def ask_llm(prompt: str) -> str:
    _client = _require_client()
    resp = _client.chat.completions.create(
        model=MODEL,
        temperature=1.0,
        messages=[
            {"role": "system", "content": "Be concise and factual."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def ask_about_water_data(
    csv_path: str,
    question: str,
    species: Optional[str] = None,
    location: Optional[str] = None,
    year: Optional[int] = None,
    parameter: Optional[str] = None,
    max_rows: int = 20,
) -> str:
    df = load_water_data(csv_path)
    df_filtered = filter_water_data(df, species=species, location=location, year=year, parameter=parameter)
    df_auto = filter_water_data_auto(df_filtered if len(df_filtered) else df, question)
    prompt = build_prompt(question, df_auto, max_rows=max_rows)
    return ask_llm(prompt)


__all__ = [
    "init_openai",
    "load_water_data",
    "filter_water_data",
    "llm_suggest_filters",
    "filter_water_data_auto",
    "build_prompt",
    "ask_llm",
    "ask_about_water_data",
]
