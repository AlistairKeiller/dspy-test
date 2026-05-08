"""GEPA-optimize a Qwen function-calling prompt against BFCL live_multiple — fully local on a Mac.

Setup (~34GB total):
    pip install -U bfcl-eval dspy
    ollama pull qwen3-coder:30b   # task model — the one we're optimizing prompts for
    ollama pull qwen3.6:27b       # reflection model — proposes the next prompt candidate
    OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=30m ollama serve

Verify plumbing without an LM:
    python bfcl_optimize.py --test

Run optimization:
    python bfcl_optimize.py

Why two different models? GEPA's reflection LM has to write *prose instructions* about
how to call tools — a meta-task. Pure code models (qwen3-coder) tend to dump JSON when
asked to write prompts, because they've been trained to emit JSON. Qwen3.6 is positioned
as a more general agentic model with strong instruction-following and built-in thinking,
which makes it more reliable at the meta-task. If reflection still produces broken
prompts, swap to the pure instruct variant: REFLECTION_LM=ollama_chat/qwen3:14b
"""

import json
import os
import random
import sys
from importlib import resources

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback


def load_examples(n=400):
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
    """First JSON object in `text`. Tolerates surrounding prose, code fences, tool tags."""
    if not text:
        return None
    i = text.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[i:])[0]
    except json.JSONDecodeError:
        return None


def judge(gold, pred):
    """Return (score, plain-English feedback). Feedback is GEPA's gradient signal —
    written without JSON syntax so the reflection LM doesn't mimic the format."""
    name, gold_args = next(iter(gold.answer[0].items()))
    call = parse(pred.tool_call or "")
    if not isinstance(call, dict):
        return 0.0, (
            "The output did not parse as a JSON object. The model must emit exactly "
            "one JSON object with a name field and an arguments field, and nothing else."
        )
    pred_name = call.get("name")
    if pred_name != name:
        offered = [t["name"] for t in json.loads(gold.tools)]
        return 0.0, (
            f"Wrong tool: the model called {pred_name} but should have called {name}. "
            f"Tools offered were: " + ", ".join(offered) + "."
        )
    args = call.get("arguments") or {}
    issues = []
    for arg, accepted in gold_args.items():
        if arg in args:
            v = args[arg]
            if v not in accepted and not any(
                str(v).lower() == str(a).lower() for a in accepted
            ):
                concrete = [a for a in accepted if a != ""]
                issues.append(
                    f"argument {arg} was set to {v!r} but should have been one of "
                    + ", ".join(repr(a) for a in concrete)
                )
        elif "" not in accepted:
            issues.append(f"missing required argument {arg}")
    if issues:
        return 0.0, f"Right tool ({name}) but: " + "; ".join(issues) + "."
    return 1.0, f"Correct call to {name}."


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    s, fb = judge(gold, pred)
    return ScoreWithFeedback(score=s, feedback=fb) if pred_name else s


def lm(model, **kw):
    """Local Ollama or any litellm-supported cloud model. think=False for Qwen3.6 stops
    the model from emitting <think>...</think> blocks that would pollute prompt outputs."""
    if model.startswith("ollama"):
        return dspy.LM(
            model, api_base="http://localhost:11434", api_key="", think=False, **kw
        )
    return dspy.LM(model, **kw)


def test():
    """Rigorous component checks. Run before optimization to verify plumbing. No LM needed."""

    class P:
        def __init__(self, t):
            self.tool_call = t

    print("1. parse()")
    cases = [
        ('{"name": "x", "arguments": {}}', {"name": "x", "arguments": {}}),
        ('Sure: {"name": "x"}', {"name": "x"}),
        ('<tool_call>\n{"name": "x"}\n</tool_call>', {"name": "x"}),
        ('```json\n{"name": "x"}\n```', {"name": "x"}),
        ("not json", None),
        ("", None),
        (None, None),
    ]
    for inp, want in cases:
        got = parse(inp)
        assert got == want, f"   ✗ parse({inp!r}) = {got!r}, expected {want!r}"
    print(f"   ✓ {len(cases)} cases")

    print("2. judge() on synthetic example")
    fake = dspy.Example(
        request="x",
        tools='[{"name": "my_tool"}]',
        answer=[{"my_tool": {"x": ["yes"], "y": ["", "default"], "z": ["only"]}}],
    )
    cases = [
        (
            "perfect",
            '{"name": "my_tool", "arguments": {"x": "yes", "z": "only"}}',
            1.0,
            "Correct",
        ),
        (
            "case-insensitive",
            '{"name": "my_tool", "arguments": {"x": "YES", "z": "only"}}',
            1.0,
            "Correct",
        ),
        (
            "optional set",
            '{"name": "my_tool", "arguments": {"x": "yes", "z": "only", "y": "default"}}',
            1.0,
            "Correct",
        ),
        (
            "optional omitted",
            '{"name": "my_tool", "arguments": {"x": "yes", "z": "only"}}',
            1.0,
            "Correct",
        ),
        ("wrong tool", '{"name": "wrong", "arguments": {}}', 0.0, "Wrong tool"),
        (
            "missing required",
            '{"name": "my_tool", "arguments": {"z": "only"}}',
            0.0,
            "missing required",
        ),
        (
            "wrong x value",
            '{"name": "my_tool", "arguments": {"x": "no", "z": "only"}}',
            0.0,
            "should have been",
        ),
        ("non-json output", "I cannot help with that", 0.0, "did not parse"),
    ]
    for desc, output, want_score, want_substr in cases:
        s, fb = judge(fake, P(output))
        assert s == want_score, f"   ✗ [{desc}] score {s} != {want_score}; fb: {fb}"
        assert want_substr.lower() in fb.lower(), (
            f"   ✗ [{desc}] missing {want_substr!r} in: {fb}"
        )
    print(f"   ✓ {len(cases)} cases")

    print("3. judge() on real BFCL example")
    examples = load_examples(n=10)
    assert len(examples) == 10
    real = examples[0]
    assert real.request and real.tools and real.answer
    gold_name, gold_args = next(iter(real.answer[0].items()))
    perfect_args = {a: v[0] for a, v in gold_args.items() if v and v[0] != ""}
    perfect_call = json.dumps({"name": gold_name, "arguments": perfect_args})
    s, fb = judge(real, P(perfect_call))
    assert s == 1.0, f"   ✗ real example: {fb}"
    print(f"   ✓ {real.request[:60]!r}…")

    print("4. metric() shape")
    import inspect

    inspect.signature(metric).bind(None, None, None, None, None)
    perfect = P('{"name": "my_tool", "arguments": {"x": "yes", "z": "only"}}')
    assert metric(fake, perfect) == 1.0
    swf = metric(fake, perfect, pred_name="select")
    assert isinstance(swf, ScoreWithFeedback) and swf.score == 1.0
    print("   ✓ float and ScoreWithFeedback paths")

    print("5. dspy.Predict(CallTool)")
    prog = dspy.Predict(CallTool)
    assert prog.signature.instructions
    assert len(list(prog.named_predictors())) == 1
    print(f"   ✓ seed prompt is {len(prog.signature.instructions)} chars")

    print("\nall tests pass ✓")


def main():
    if "--test" in sys.argv:
        test()
        return

    task = lm(os.getenv("TASK_LM", "ollama_chat/qwen3-coder:30b"), temperature=0.0)
    refl = lm(
        os.getenv("REFLECTION_LM", "ollama_chat/qwen3.6:27b"),
        temperature=1.0,
        max_tokens=8192,
    )
    dspy.configure(lm=task)

    examples = load_examples()
    random.Random(42).shuffle(examples)
    train, val, test_set = examples[:80], examples[80:230], examples[230:380]

    prog = dspy.Predict(CallTool)
    score = lambda p: (
        dspy.Evaluate(devset=test_set, metric=metric, num_threads=4)(p).score
    )

    print(f"baseline:  {score(prog):.1f}%")
    optimized = dspy.GEPA(metric=metric, reflection_lm=refl, auto="light").compile(
        prog, trainset=train, valset=val
    )
    print(f"optimized: {score(optimized):.1f}%")
    print(f"\n── Optimized prompt ──\n{optimized.signature.instructions}")


if __name__ == "__main__":
    main()
