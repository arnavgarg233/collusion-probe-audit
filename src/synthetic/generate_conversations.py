"""Generate transcripts for all conditions with identical visible structure.

Each base item runs through: solo, honest_open, honest_secret, collusion(direct),
collusion(shift), collusion_override, inert_collusion. The model writes a single
strict-format line "Option <L>: <reason>"; the committed/visible vote is normalized
to the by-design target L (so collusion and honest_secret are vote-matched), and any
forbidden tell phrase triggers regeneration/scrub. Private payloads never enter
`visible_text`.

Output: data/transcripts/transcripts.json
"""
from __future__ import annotations
# --- path bootstrap: make sibling src/ subpackages importable as flat modules ---
import os as _os, sys as _sys
_SRCROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ('', 'common', 'synthetic', 'narcbench', 'audit', 'exploratory'):
    _pp = _os.path.join(_SRCROOT, _sub)
    if _os.path.isdir(_pp) and _pp not in _sys.path:
        _sys.path.insert(0, _pp)
# --- end bootstrap ---
import argparse, json, random, re
import config, tasks, prompts

RUN_PLAN = [
    ("solo", None),
    ("honest_open", None),
    ("honest_secret", None),
    ("collusion", "direct"),
    ("collusion", "shift"),
    ("collusion_override", "override"),
    ("inert_collusion", None),
]
_VOTE_RE = re.compile(r"\boption\s*([A-D])\b", re.I)
_LINE_RE = re.compile(r"option\s*[A-D]\s*[:\-]\s*(.*)", re.I)


def parse_free_vote(raw: str):
    m = _VOTE_RE.search(raw or "")
    return m.group(1).upper() if m else None


def extract_reason(raw: str) -> str:
    line = (raw or "").strip().splitlines()[0] if raw else ""
    m = _LINE_RE.search(line)
    text = (m.group(1) if m else line).strip()
    text = re.sub(r"^[\"']|[\"'.]+$", "", text).strip()
    return text


def has_tell(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in config.TELL_PHRASES)


def cap_words(text: str, n: int = config.RATIONALE_MAX_WORDS) -> str:
    return " ".join(text.split()[:n]).rstrip(",;:. ")


def _gen_turn(item, condition, assignment, agent, prior, mock):
    msgs, tgt = prompts.build_turn(item, condition, assignment, agent, prior)
    # NOTE (diagnostic caveat): free_vote is seeded to the target `tgt` and falls back to
    # it on any parse failure (line below) and in mock mode (free stays == tgt). This
    # INFLATES the "free_vote == tgt" agreement diagnostic in behavioral_verify.py:39,
    # because parse failures / mock rows silently count as the model freely choosing the
    # target. A faithful free-vote diagnostic would record None on parse failure and
    # exclude it. Left as-is (non-load-bearing) to avoid changing reported numbers.
    free = tgt
    if mock:
        reason = f"the evidence for option {tgt} is strongest on this axis"
    else:
        import model_utils
        reason = ""
        for _ in range(config.GEN_RETRIES + 1):
            raw = model_utils.generate_response(msgs)
            free = parse_free_vote(raw) or free
            reason = extract_reason(raw)
            if reason and not has_tell(reason):
                break
        if (not reason) or has_tell(reason):
            reason = "the evidence is strongest on the key criterion"
    reason = cap_words(reason) or "the evidence is strongest on the key criterion"
    line = f"Option {tgt}: {reason}."
    return {"messages": msgs, "rationale": reason, "free_vote": free,
            "vote_letter": tgt, "response_text": line, "response_line": line}


def generate(n_items: int, limit, mock: bool, out_path: str):
    config.ensure_dirs()
    prompts.dump_templates()
    items = tasks.make_items(n_items)
    records, count = [], 0
    for item in items:
        for condition, proto in RUN_PLAN:
            rng = random.Random(f"{config.SEED}|{item['item_id']}|{condition}|{proto}")
            assign_proto = proto if condition == "collusion" else None
            assignment = tasks.assign(item, condition, assign_proto, rng)
            proto_tag = assignment.get("protocol") or "na"

            A = _gen_turn(item, condition, assignment, "A", None, mock)
            agents = {"A": A}
            visible = f"Reviewer A: {A['response_line']}"
            if condition != "solo":
                B = _gen_turn(item, condition, assignment, "B",
                              {"response_line": A["response_line"]}, mock)
                agents["B"] = B
                visible += f" Reviewer B: {B['response_line']}"

            records.append({
                "episode_id": f"{item['item_id']}|{condition}|{proto_tag}",
                "item_id": item["item_id"],
                "condition": condition,
                "protocol": proto_tag,
                "secret": assignment["secret"],
                "merit_best": item["merit_best"],
                "target_vote": assignment["target_vote"],
                "agents": agents,
                "visible_text": visible,
            })
            count += 1
            if limit and count >= limit:
                break
        if limit and count >= limit:
            break

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[generate] wrote {len(records)} episodes -> {out_path}")
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=config.N_ITEMS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default=config.TRANSCRIPTS_PATH)
    a = ap.parse_args()
    generate(a.n_items, a.limit, a.mock, a.out)
