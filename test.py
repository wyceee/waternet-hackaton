"""Simple CLI test harness using the shared core functions.

Run with: python test.py "Your question"
"""

from water_core import init_openai, ask_about_water_data
import os, argparse


def load_key() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if os.path.exists("openai.key"):
        with open("openai.key") as f:
            return f.read().strip()
    raise SystemExit("No API key found. Set OPENAI_API_KEY or create openai.key")


def main():
    parser = argparse.ArgumentParser(description="Ask questions about a water quality CSV")
    parser.add_argument("question", nargs="*", help="Question to ask (if empty sample questions are used)")
    parser.add_argument("--csv", default="HB100sampled.csv", help="Path to semicolon separated CSV")
    args = parser.parse_args()
    api_key = load_key()
    init_openai(api_key)
    questions = args.question or [
        "What species were found at volkstuinen Amstelglorie?",
        "How many different species were observed in 1998?",
    ]
    for q in questions:
        print("-" * 60)
        print(f"Q: {q}")
        try:
            answer = ask_about_water_data(args.csv, q)
            print(f"A: {answer}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
