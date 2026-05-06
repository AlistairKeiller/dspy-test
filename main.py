import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

HF = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
DATA = Path(".bfcl_data")
FILES = {
    "mq": "BFCL_v3_live_multiple.json",
    "ma": "possible_answer/BFCL_v3_live_multiple.json",
    "iq": "BFCL_v3_live_irrelevance.json",
}


def fetch(remote, local):
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {remote}")
        urllib.request.urlretrieve(f"{HF}/{remote}", local)


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def load_splits(train_n=60, val_n=30, test_n=150, irrel_frac=0.3, seed=42):
    DATA.mkdir(exist_ok=True)
    for r in FILES.values():
        fetch(r, DATA / r)
    mult = load_jsonl(DATA / FILES["mq"])
    answers = {r["id"]: r["ground_truth"] for r in load_jsonl(DATA / FILES["ma"])}
    irrel = load_jsonl(DATA / FILES["iq"])

    def row(q, gt):
        text = "\n".join(
            m["content"] for m in q["question"][0] if m.get("role") == "user"
        )
        return {"request": text, "tools": q["function"], "ground_truth": gt}

    rng = random.Random(seed)
    rng.shuffle(mult)
    rng.shuffle(irrel)
    n = train_n + val_n + test_n
    n_irrel = int(irrel_frac * n)
    pool = [row(q, answers.get(q["id"], [])) for q in mult[: n - n_irrel]] + [
        row(q, []) for q in irrel[:n_irrel]
    ]
    rng.shuffle(pool)
    examples = [
        dspy.Example(
            request=r["request"],
            tools=json.dumps(r["tools"], separators=(",", ":")),
            ground_truth=json.dumps(r["ground_truth"], separators=(",", ":")),
        ).with_inputs("request", "tools")
        for r in pool
    ]
    return (
        examples[:train_n],
        examples[train_n : train_n + val_n],
        examples[train_n + val_n :],
    )


class SelectToolCall(dspy.Signature):
    """Decide whether and how to call exactly one tool to fulfill the user's request.

    Rules:
    - If a tool can directly answer the request, call it with arguments drawn from the request.
    - If NO available tool addresses the request, return {"name": null, "arguments": {}}.
    - At most ONE tool call. Never invent tools that are not in the list.
    - Output ONLY a single JSON object — no prose, no code fences.
    """

    request: str = dspy.InputField()
    tools: str = dspy.InputField(desc="JSON list of available tool schemas.")
    tool_call: str = dspy.OutputField(desc='{"name": <tool|null>, "arguments": {...}}')


class ToolUseProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.select = dspy.Predict(SelectToolCall)

    def forward(self, request, tools):
        return self.select(request=request, tools=tools)


def parse_call(raw):
    if not raw:
        return None
    s = raw.strip().strip("`")
    if s.lower().startswith("json"):
        s = s[4:].strip()
    if "{" in s and not s.startswith("{"):
        s = s[s.find("{") :]
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("arguments") or obj.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    if isinstance(name, str) and name.lower() == "null":
        name = None
    return {"name": name if isinstance(name, str) else None, "arguments": args}


def matches(pred, accepted):
    if pred in accepted:
        return True
    if isinstance(pred, (int, float)):
        return any(
            isinstance(a, (int, float)) and abs(pred - a) < 1e-6 for a in accepted
        )
    if isinstance(pred, str):
        p = pred.lower().strip()
        return any(isinstance(a, str) and a.lower().strip() == p for a in accepted)
    if isinstance(pred, list):
        return any(
            isinstance(a, list)
            and len(a) == len(pred)
            and all(matches(x, [y]) for x, y in zip(pred, a))
            for a in accepted
        )
    return False


def score_call(pred, gold):
    name, args = pred["name"], pred["arguments"]
    if not gold:
        if name is None:
            return 1.0, "Correctly abstained — no tool fits this request."
        return 0.0, (
            f"Over-called `{name}`: none of the available tools answer this. "
            'When intent does not map to a tool, output {"name": null, "arguments": {}}.'
        )
    gold_name, gold_args = next(iter(gold[0].items()))
    if name is None:
        return 0.0, f"Under-called: should have invoked `{gold_name}`."
    if name != gold_name:
        return 0.0, f"Wrong tool: predicted `{name}`, expected `{gold_name}`."
    required = [a for a, acc in gold_args.items() if "" not in acc]
    optional = [a for a in gold_args if a not in required]
    correct, issues = 0, []
    for a in required:
        if a not in args:
            issues.append(f"missing required `{a}`")
        elif matches(args[a], gold_args[a]):
            correct += 1
        else:
            issues.append(f"`{a}`={args[a]!r} expected one of {gold_args[a]!r}")
    for a in optional:
        if a in args and matches(args[a], gold_args[a]):
            correct += 1
    extra = [a for a in args if a not in gold_args]
    if extra:
        issues.append(f"unrequested args: {extra}")
    total = len(required) + len(optional)
    score = 0.5 + 0.5 * (correct / total if total else 1.0)
    if score == 1.0 and not issues:
        return 1.0, f"Perfect call to `{gold_name}`."
    return score, (
        f"Right tool (`{gold_name}`), but: "
        + "; ".join(issues)
        + ". Extract values directly from the request, match types, "
        "and don't add args the user didn't specify."
    )


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    call = parse_call(getattr(pred, "tool_call", None))
    if call is None:
        s, fb = (
            0.0,
            (
                "Output was not valid JSON. Always emit a single JSON object: "
                '{"name": <tool|null>, "arguments": {...}} with no prose or fences.'
            ),
        )
    else:
        s, fb = score_call(call, json.loads(gold.ground_truth))
    return ScoreWithFeedback(score=s, feedback=fb) if pred_name else s


def make_lm(model, **kw):
    return dspy.LM(
        f"ollama_chat/{model}",
        api_base=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        api_key="",
        **kw,
    )


def reflection_lm():
    if os.getenv("ANTHROPIC_API_KEY"):
        print("→ reflection: Claude")
        return dspy.LM("anthropic/claude-opus-4-5", temperature=1.0, max_tokens=32000)
    if os.getenv("OPENAI_API_KEY"):
        print("→ reflection: GPT")
        return dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=32000)
    print(
        "→ reflection: local gemma4:31b (set ANTHROPIC_API_KEY or OPENAI_API_KEY for speed)"
    )
    return make_lm("gemma4:31b", temperature=1.0, max_tokens=8000)


def evaluate(prog, data, threads=8):
    return dspy.Evaluate(
        devset=data, metric=metric, num_threads=threads, display_progress=True
    )(prog)


def main():
    mini = "--mini" in sys.argv
    budget = next(
        (a for a in ("light", "medium", "heavy") if f"--{a}" in sys.argv), "light"
    )
    task_model = "gemma4:e4b" if mini else "gemma4:31b"
    print(f"→ task: {task_model}    budget: {budget}")

    dspy.configure(lm=make_lm(task_model, temperature=0.0, max_tokens=512))
    train, val, test = load_splits()
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

    prog = ToolUseProgram()
    print("\nbaseline...")
    t = time.time()
    base = evaluate(prog, test)
    print(f"  {base:.3f}  ({time.time() - t:.0f}s)")

    print("\nGEPA optimizing...")
    t = time.time()
    optimized = dspy.GEPA(
        metric=metric, reflection_lm=reflection_lm(), auto=budget
    ).compile(prog, trainset=train, valset=val)
    print(f"  done ({time.time() - t:.0f}s)")

    print("\noptimized...")
    t = time.time()
    opt = evaluate(optimized, test)
    print(f"  {opt:.3f}  ({time.time() - t:.0f}s)")

    out = Path("optimized_program")
    out.mkdir(exist_ok=True)
    optimized.save(str(out / "program.json"))
    summary = {
        "baseline": round(base, 4),
        "optimized": round(opt, 4),
        "delta": round(opt - base, 4),
        "instruction": optimized.select.signature.instructions,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nbaseline {base:.3f} → optimized {opt:.3f}  (Δ {opt - base:+.3f})")
    print("─" * 70 + "\n" + summary["instruction"] + "\n" + "─" * 70)


if __name__ == "__main__":
    main()
