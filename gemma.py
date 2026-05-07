import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

GH = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data"
DATA = Path(".bfcl_v4_data")
FILES = {
    "mq": "BFCL_v4_live_multiple.json",
    "ma": "possible_answer/BFCL_v4_live_multiple.json",
    "rq": "BFCL_v4_live_relevance.json",
    "iq": "BFCL_v4_live_irrelevance.json",
}
MAX_TOOLS_CHARS = 6000


def fetch(remote, local):
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"  ↓ {remote}")
        urllib.request.urlretrieve(f"{GH}/{remote}", local)


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def make_rows(qs, answers=None, synth_truth=False):
    out = []
    for q in qs:
        msgs = q["question"][0]
        text = "\n".join(m["content"] for m in msgs if m.get("role") == "user")
        tools_json = json.dumps(q["function"], separators=(",", ":"))
        if len(tools_json) > MAX_TOOLS_CHARS:
            continue
        if answers is not None:
            gt = answers.get(q["id"], [])
        elif synth_truth and q["function"]:
            gt = [{q["function"][0]["name"]: {}}]
        else:
            gt = []
        out.append({"request": text, "tools": q["function"], "ground_truth": gt})
    return out


def to_examples(rows):
    return [
        dspy.Example(
            request=r["request"],
            tools=json.dumps(r["tools"], separators=(",", ":")),
            ground_truth=json.dumps(r["ground_truth"], separators=(",", ":")),
        ).with_inputs("request", "tools")
        for r in rows
    ]


def load_splits(train_n=80, val_n=50, test_n=150, seed=42):
    DATA.mkdir(exist_ok=True)
    for r in FILES.values():
        fetch(r, DATA / r)
    rng = random.Random(seed)

    mult = make_rows(
        load_jsonl(DATA / FILES["mq"]),
        {r["id"]: r["ground_truth"] for r in load_jsonl(DATA / FILES["ma"])},
    )
    rel = make_rows(load_jsonl(DATA / FILES["rq"]), synth_truth=True)
    irrel = make_rows(load_jsonl(DATA / FILES["iq"]))
    rng.shuffle(mult)
    rng.shuffle(rel)
    rng.shuffle(irrel)

    n = train_n + val_n + test_n
    n_irrel, n_mult = n // 2, int(n * 0.35)
    n_rel = n - n_irrel - n_mult
    pool = mult[:n_mult] + rel[:n_rel] + irrel[:n_irrel]
    rng.shuffle(pool)
    examples = to_examples(pool)
    return (
        examples[:train_n],
        examples[train_n : train_n + val_n],
        examples[train_n + val_n : n],
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
    if s.endswith("```"):
        s = s[:-3].rstrip()
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


def score_call(pred, gold, tools):
    name, args = pred["name"], pred["arguments"]
    if not gold:
        if name is None:
            return 1.0, "Correctly abstained — no tool fits this request."
        return 0.0, (
            f"Over-called `{name}`: none of the available tools fit this request. "
            'When the user\'s intent does not clearly map to a tool, output {"name": null, "arguments": {}}. '
            "Topical proximity is not the same as fit — read each description literally."
        )
    gold_name, gold_args = next(iter(gold[0].items()))
    if name is None:
        return (
            0.0,
            f"Under-called: should have invoked `{gold_name}`. The user's request maps to this tool.",
        )
    if name != gold_name:
        gold_desc = next(
            (t.get("description", "") for t in tools if t.get("name") == gold_name), ""
        )
        return 0.0, (
            f"Wrong tool: predicted `{name}`, expected `{gold_name}` — for: {gold_desc[:180]}. "
            "Pick the tool whose stated purpose most directly matches the request."
        )
    if not gold_args:
        return 1.0, f"Correctly invoked `{gold_name}`."
    required = [a for a, acc in gold_args.items() if "" not in acc]
    optional = [a for a in gold_args if a not in required]
    correct, issues = 0, []
    for a in required:
        if a not in args:
            issues.append(f"missing required `{a}`")
        elif matches(args[a], gold_args[a]):
            correct += 1
        else:
            ref = next((v for v in gold_args[a] if v != ""), None)
            kind = (
                "wrong type"
                if ref is not None and type(args[a]) is not type(ref)
                else "wrong value"
            )
            issues.append(
                f"`{a}`={args[a]!r} ({kind}); expected one of {gold_args[a]!r}"
            )
    for a in optional:
        if a not in args:
            correct += 1
        elif matches(args[a], gold_args[a]):
            correct += 1
        else:
            issues.append(
                f"optional `{a}`={args[a]!r} is wrong; expected one of {gold_args[a]!r} or omitted"
            )
    extra = [a for a in args if a not in gold_args]
    if extra:
        issues.append(
            f"unrequested args {extra} — only include args the user specified"
        )
    total = len(required) + len(optional)
    score = 0.5 + 0.5 * (correct / total if total else 1.0)
    if score == 1.0 and not issues:
        return 1.0, f"Perfect call to `{gold_name}`."
    return score, (
        f"Right tool (`{gold_name}`), but: "
        + "; ".join(issues)
        + ". Extract values directly from the request and match expected types exactly."
    )


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    call = parse_call(getattr(pred, "tool_call", None))
    if call is None:
        s, fb = (
            0.0,
            (
                "Output was not valid JSON. Always emit a single JSON object: "
                '{"name": <tool|null>, "arguments": {...}} with no prose, fences, or trailing commas.'
            ),
        )
    else:
        s, fb = score_call(call, json.loads(gold.ground_truth), json.loads(gold.tools))
    return ScoreWithFeedback(score=s, feedback=fb) if pred_name else s


def make_lm(model, **kw):
    return dspy.LM(
        f"ollama_chat/{model}",
        api_base=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        api_key="",
        **kw,
    )


def reflection_lm():
    model = os.getenv("REFLECTION_MODEL", "gemma4:e4b")
    print(f"→ reflection: {model}")
    return make_lm(model, temperature=1.0, max_tokens=8192)


def evaluate(prog, data, threads):
    return dspy.Evaluate(
        devset=data, metric=metric, num_threads=threads, display_progress=True
    )(prog)


def main():
    mini = "--mini" in sys.argv
    budget = next(
        (a for a in ("light", "medium", "heavy") if f"--{a}" in sys.argv), "light"
    )
    threads = int(
        next(
            (sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--threads"),
            4,
        )
    )
    task_model = os.getenv("TASK_MODEL", "qwen3-coder:30b" if mini else "gemma4:e4b")
    print(f"→ task: {task_model}    budget: {budget}    threads: {threads}")
    print(
        f"  tip: export OLLAMA_NUM_PARALLEL={threads} OLLAMA_KEEP_ALIVE=30m before starting Ollama."
    )

    dspy.configure(lm=make_lm(task_model, temperature=0.0, max_tokens=512), cache=True)
    train, val, test = load_splits()
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

    prog = ToolUseProgram()
    print("\nbaseline...")
    t = time.time()
    base = evaluate(prog, test, threads)
    print(f"  {base}  ({time.time() - t:.0f}s)")

    print("\nGEPA optimizing...")
    t = time.time()
    optimized = dspy.GEPA(
        metric=metric, reflection_lm=reflection_lm(), auto=budget
    ).compile(prog, trainset=train, valset=val)
    print(f"  done ({time.time() - t:.0f}s)")

    print("\noptimized...")
    t = time.time()
    opt = evaluate(optimized, test, threads)
    print(f"  {opt}  ({time.time() - t:.0f}s)")

    out = Path("optimized_program")
    out.mkdir(exist_ok=True)
    optimized.save(str(out / "program.json"))
    instructions = {
        n: p.signature.instructions for n, p in optimized.named_predictors()
    }
    summary = {
        "bfcl_version": "v4",
        "task_model": task_model,
        "predictor": "Predict",
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "baseline": round(base.score, 4),
        "optimized": round(opt.score, 4),
        "delta": round(opt.score - base.score, 4),
        "instructions": instructions,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nbaseline {base} → optimized {opt}  (Δ {opt.score - base.score})")
    for name, instr in instructions.items():
        print(f"\n── {name} " + "─" * max(0, 66 - len(name)))
        print(instr)


if __name__ == "__main__":
    main()
