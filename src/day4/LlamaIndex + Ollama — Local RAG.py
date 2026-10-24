
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parents[2]
OUTPUT = BASE / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

API_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "You are a helpful AI and robotics tutor. "
    "Explain concepts using simple English, short answers, "
    "and practical examples suitable for a BTech student."
)

def ask_ollama(model, prompt, system_prompt=SYSTEM_PROMPT):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.3}
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))

        elapsed = round(time.perf_counter() - start, 2)

        return {
            "model": model,
            "question": prompt,
            "answer": result["message"]["content"],
            "response_time_seconds": elapsed,
            "error": ""
        }

    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
        return {
            "model": model,
            "question": prompt,
            "answer": "",
            "response_time_seconds": None,
            "error": str(error)
        }

def main():
    results = []

    test_prompts = [
        "What is artificial intelligence? Give 3 points.",
        "Explain machine learning with one example.",
        "What is sensor fusion in robotics?",
        "How does a camera help an autonomous vehicle?",
        "Explain the difference between a robot sensor and an actuator."
    ]

    print("Testing llama3.2:3b with 5 prompts")

    for number, prompt in enumerate(test_prompts, start=1):
        print(f"\nPrompt {number}: {prompt}")
        result = ask_ollama("llama3.2:3b", prompt)
        results.append(result)

        if result["error"]:
            print("Error:", result["error"])
        else:
            print("Answer:", result["answer"])
            print("Response time:", result["response_time_seconds"], "seconds")

    comparison_questions = [
        "Explain sensor fusion in simple words.",
        "How can AI help a robot avoid obstacles?",
        "What is the difference between supervised and unsupervised learning?"
    ]

    print("\nComparing llama3.2:3b and qwen2.5:3b")

    for question in comparison_questions:
        for model in ("llama3.2:3b", "qwen2.5:3b"):
            print(f"\nModel: {model} | Question: {question}")
            result = ask_ollama(model, question)
            results.append(result)

            if result["error"]:
                print("Error:", result["error"])
            else:
                print("Answer:", result["answer"])
                print("Response time:", result["response_time_seconds"], "seconds")

    output_file = OUTPUT / "w7d4_results.json"
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "system_prompt": SYSTEM_PROMPT,
        "results": results
    }

    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nSaved all results to: {output_file}")

if __name__ == "__main__":
    main()
