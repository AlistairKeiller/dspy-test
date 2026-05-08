import json
import os
import random
import re
import sys
import time
import types
from collections import Counter
from importlib import resources
from pathlib import Path

# bfcl-eval's MODEL_CONFIG_MAPPING transitively imports every model SDK on the
# planet (qwen-agent, cohere, anthropic, ...). Stub it before importing the checker.
_stub = types.ModuleType("bfcl_eval.constants.model_config")


class _DefaultCfg:
    underscore_to_dot = False


class _MapAny(dict):
    def __getitem__(self, k):
        return _DefaultCfg()

    def __contains__(self, k):
        return True


_stub.MODEL_CONFIG_MAPPING = _MapAny()
sys.modules["bfcl_eval.constants.model_config"] = _stub

import dspy
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback


def _resolve_lang():
    for path in (
        "bfcl_eval.constants.enums",
        "bfcl_eval.eval_checker.ast_eval.ast_checker",
    ):
        try:
            return __import__(path, fromlist=["Language"]).Language.PYTHON
        except (ImportError, AttributeError):
            pass
    return "python"


PYTHON_LANG = _resolve_lang()


CATEGORIES = ["live_simple", "live_multiple", "live_relevance", "live_irrelevance"]
TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
NULL_CALL = '{"name": null, "arguments": {}}'


def load_examples():
    base = resources.files("bfcl_eval") / "data"
    rows = []
    for cat in CATEGORIES:
        qs = [
            json.loads(l)
            for l in (base / f"BFCL_v4_{cat}.json").read_text().splitlines()
            if l.strip()
        ]
        try:
            ans_path = base / "possible_answer" / f"BFCL_v4_{cat}.json"
            answers = {
                r["id"]: r["ground_truth"]
                for r in (
                    json.loads(l)
                    for l in ans_path.read_text().splitlines()
                    if l.strip()
                )
            }
        except FileNotFoundError:
            answers = None
        for q in qs:
            text = "\n".join(
                m["content"]
                for m in q["question"][0]
                if m.get("role") in {"user", "system"}
            )
            rows.append(
                dspy.Example(
                    request=text,
                    tools=json.dumps(q["function"], separators=(",", ":")),
                    gold_json=json.dumps(answers[q["id"]]) if answers else "null",
                    category=cat,
                ).with_inputs("request", "tools")
            )
    return rows


class ToolApplies(dspy.Signature):
    """Decide whether any of the offered tools can directly fulfill the user's request.

    A tool 'applies' only when the user's stated intent maps onto the tool's stated purpose.
    Topical proximity does NOT count: a tool about X cannot do Y just because both share a domain.
    """

    request: str = dspy.InputField()
    tools: str = dspy.InputField(desc="JSON list of available tool schemas.")
    applies: bool = dspy.OutputField(
        desc="True iff at least one offered tool can fulfill the request."
    )


class CallTool(dspy.Signature):
    """Pick exactly one offered tool and emit a single JSON call.

    Extract argument values verbatim from the request — never invent or modify values.
    Match expected types exactly (string vs integer vs boolean vs list).
    Output ONLY the JSON object, no prose, no code fences.
    """

    request: str = dspy.InputField()
    tools: str = dspy.InputField(desc="JSON list of available tool schemas.")
    tool_call: str = dspy.OutputField(desc='{"name": <tool>, "arguments": {...}}')


class ToolUseProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.applies = dspy.Predict(ToolApplies)
        self.call = dspy.Predict(CallTool)

    def forward(self, request, tools):
        if not self.applies(request=request, tools=tools).applies:
            return dspy.Prediction(tool_call=NULL_CALL)
        return self.call(request=request, tools=tools)


def parse(raw):
    if not raw:
        return None
    s = (m.group(1) if (m := TAG.search(raw)) else raw).strip()
    s = s.lstrip("`").lstrip()
    if s.lower().startswith("json"):
        s = s[4:].lstrip()
    i = next((p for p in (s.find("{"), s.find("[")) if p >= 0), -1)
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[i:])
    except Exception:
        return None
    if isinstance(obj, list):
        obj = obj[0] if obj else None
    if not isinstance(obj, dict):
        return None
    raw_name = obj.get("name")
    name = (
        raw_name if isinstance(raw_name, str) and raw_name.lower() != "null" else None
    )
    args = next(
        (
            a
            for a in (obj.get("arguments"), obj.get("args"), obj.get("parameters"))
            if isinstance(a, dict)
        ),
        {},
    )
    return name, args


def _score(gold, pred):
    """(score: 0.0 or 1.0, feedback: str). Feedback embeds the request and offered
    tools so GEPA's reflection LM can learn failure-mode patterns, not just rules."""
    parsed = parse(getattr(pred, "tool_call", None))
    cat = gold.category
    tools = json.loads(gold.tools)
    req = gold.request[:240].replace("\n", " ")
    offered = [t["name"] for t in tools]

    if cat == "live_irrelevance":
        if not parsed or parsed[0] is None:
            return 1.0, "Correctly abstained — no offered tool fits."
        return 0.0, (
            f'OVER-CALLED on: "{req}"\n'
            f"You called `{parsed[0]}`, but none of the offered tools {offered} "
            f"actually address this request. Read each tool description literally — "
            f"topical proximity to a tool name is NOT the same as that tool fitting the request."
        )

    if cat == "live_relevance":
        if parsed and parsed[0] in offered:
            return 1.0, f"Correctly invoked `{parsed[0]}`."
        return 0.0, (
            f'UNDER-CALLED on: "{req}"\n'
            f"At least one of the offered tools {offered} fits this request — pick the most relevant."
        )

    # AST: live_simple, live_multiple
    if not parsed or parsed[0] is None:
        return 0.0, (
            f'NO CALL on: "{req}"\nThis request requires invoking one of: {offered}.'
        )
    name, args = parsed
    result = ast_checker(
        tools, [{name: args}], json.loads(gold.gold_json), PYTHON_LANG, cat, "any-model"
    )
    if result["valid"]:
        return 1.0, f"Correct call to `{name}`."
    errs = "; ".join(result.get("error", []))[:400]
    return 0.0, (
        f'WRONG CALL on: "{req}"\n'
        f"You emitted: {name}({args})\n"
        f"BFCL official check: {errs}"
    )


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    s, fb = _score(gold, pred)
    return ScoreWithFeedback(score=s, feedback=fb) if pred_name else s


def make_lm(model, **kw):
    return dspy.LM(
        f"ollama_chat/{model}",
        api_base=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        api_key="",
        **kw,
    )


def per_category(results):
    by = {}
    for ex, _, sc in results:
        v = sc if isinstance(sc, (int, float)) else getattr(sc, "score", 0)
        by.setdefault(ex.category, []).append(v)
    return {c: round(100 * sum(v) / len(v), 2) for c, v in by.items()}


def split_balanced(rows, train_n, val_n, test_n, seed=42):
    rng = random.Random(seed)
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r.category, []).append(r)
    for rs in by_cat.values():
        rng.shuffle(rs)
    n_each = (train_n + val_n + test_n) // len(by_cat) + 1
    pool = [r for cat in by_cat for r in by_cat[cat][:n_each]]
    rng.shuffle(pool)
    return (
        pool[:train_n],
        pool[train_n : train_n + val_n],
        pool[train_n + val_n : train_n + val_n + test_n],
    )


def main():
    budget = next(
        (a for a in ("light", "medium", "heavy") if f"--{a}" in sys.argv), "light"
    )
    threads = int(os.getenv("THREADS", "4"))
    task_model = os.getenv("TASK_MODEL", "qwen3-coder:30b")
    refl_model = os.getenv("REFLECTION_MODEL", "qwen3-coder:30b")
    print(
        f"→ task: {task_model}    reflection: {refl_model}    budget: {budget}    threads: {threads}"
    )

    dspy.configure(lm=make_lm(task_model, temperature=0.0, max_tokens=512), cache=True)
    train, val, test = split_balanced(
        load_examples(), train_n=100, val_n=100, test_n=200
    )
    print(
        f"  train={len(train)}  val={len(val)}  test={len(test)}  {dict(Counter(e.category for e in test))}"
    )

    prog = ToolUseProgram()

    print("\nbaseline...")
    t = time.time()
    base = dspy.Evaluate(
        devset=test, metric=metric, num_threads=threads, display_progress=True
    )(prog)
    print(
        f"  {base.score:.2f}  per-cat={per_category(base.results)}  ({time.time() - t:.0f}s)"
    )

    print("\nGEPA optimizing...")
    t = time.time()
    optimized = dspy.GEPA(
        metric=metric,
        reflection_lm=make_lm(refl_model, temperature=1.0, max_tokens=8192),
        auto=budget,
    ).compile(prog, trainset=train, valset=val)
    print(f"  done ({time.time() - t:.0f}s)")

    print("\noptimized...")
    t = time.time()
    opt = dspy.Evaluate(
        devset=test, metric=metric, num_threads=threads, display_progress=True
    )(optimized)
    print(
        f"  {opt.score:.2f}  per-cat={per_category(opt.results)}  ({time.time() - t:.0f}s)"
    )

    out = Path("optimized_program")
    out.mkdir(exist_ok=True)
    optimized.save(str(out / "program.json"))
    instructions = {
        n: p.signature.instructions for n, p in optimized.named_predictors()
    }
    summary = {
        "baseline": round(base.score, 2),
        "optimized": round(opt.score, 2),
        "delta": round(opt.score - base.score, 2),
        "baseline_per_category": per_category(base.results),
        "optimized_per_category": per_category(opt.results),
        "instructions": instructions,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"\nbaseline {base.score:.2f} → optimized {opt.score:.2f}  (Δ {opt.score - base.score:+.2f})"
    )
    for name, instr in instructions.items():
        print(f"\n── {name} " + "─" * max(0, 66 - len(name)))
        print(instr)


if __name__ == "__main__":
    main()
