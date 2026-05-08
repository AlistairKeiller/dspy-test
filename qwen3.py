"""GEPA-optimize a Qwen function-calling prompt against BFCL live_multiple.

ollama pull qwen3-coder:30b
OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=30m ollama serve
uv run qwen3.py
"""

import json
import random
from importlib import resources

import dspy


def load_examples(n=300):
    """BFCL v4 live_multiple: each request offers 2-4 tools; pick the right one with right args."""
    base = resources.files("bfcl_eval") / "data"
    answers = {
        r["id"]: r["ground_truth"]
        for r in (
            json.loads(l)
            for l in (base / "possible_answer/BFCL_v4_live_multiple.json")
            .read_text()
            .splitlines()
        )
    }
    examples = []
    for line in (base / "BFCL_v4_live_multiple.json").read_text().splitlines()[:n]:
        q = json.loads(line)
        examples.append(
            dspy.Example(
                request=q["question"][0][0]["content"],
                tools=json.dumps(q["function"]),
                answer=answers[q["id"]],
            ).with_inputs("request", "tools")
        )
    return examples


class CallTool(dspy.Signature):
    """Pick exactly one tool from the offered list and emit a JSON call.
    Output ONLY a JSON object: {"name": ..., "arguments": {...}}."""

    request: str = dspy.InputField()
    tools: str = dspy.InputField()
    tool_call: str = dspy.OutputField()


def parse(text):
    """First JSON object in `text`. Tolerates surrounding prose, code fences, tool-call tags."""
    i = text.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[i:])[0]
    except json.JSONDecodeError:
        return None


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """1.0 iff the call matches BFCL's possible-answer: right tool, required args present, accepted values."""
    call = parse(pred.tool_call or "")
    if not isinstance(call, dict):
        return 0.0
    name, gold_args = next(iter(gold.answer[0].items()))
    if call.get("name") != name:
        return 0.0
    args = call.get("arguments") or {}
    for arg, accepted in gold_args.items():
        if arg in args:
            v = args[arg]
            if v not in accepted and not any(
                str(v).lower() == str(a).lower() for a in accepted
            ):
                return 0.0
        elif "" not in accepted:  # "" in accepted means this arg may be omitted
            return 0.0
    return 1.0


def main():
    lm = dspy.LM(
        "ollama_chat/qwen3-coder:30b", api_base="http://localhost:11434", api_key=""
    )
    dspy.configure(lm=lm)

    examples = load_examples()
    random.Random(42).shuffle(examples)
    train, val, test = examples[:80], examples[80:130], examples[130:]

    prog = dspy.Predict(CallTool)
    score = lambda p: dspy.Evaluate(devset=test, metric=metric, num_threads=4)(p).score

    print(f"baseline:  {score(prog):.1f}%")
    optimized = dspy.GEPA(metric=metric, reflection_lm=lm, auto="light").compile(
        prog, trainset=train, valset=val
    )
    print(f"optimized: {score(optimized):.1f}%")
    print(f"\n── Optimized prompt ──\n{optimized.signature.instructions}")


if __name__ == "__main__":
    main()
