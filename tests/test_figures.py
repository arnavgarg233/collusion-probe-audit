"""The values drawn in Figs. 1-2 agree with the stored records."""

from collusion_probe_audit import derive, figures

FIG_TO_SETTING = {
    "gpt-oss core": "narcbench_gptoss_core", "Qwen3 core": "narcbench_qwen3_core",
    "Liars' convincing": "liars_convincing_game", "Liars' gender": "liars_gender_secret",
    "sleeper-agent": "sleeper_agent", "BackdoorLLM": "backdoorllm", "b2w ai_liar": "b2w_ai_liar",
    "b2w dilution": "b2w_deception_dilution", "b2w insider": "b2w_insider_trading",
    "b2w sandbagging": "b2w_sandbagging", "b2w apps_backdoor": "b2w_apps_backdoors",
}


def test_drawn_deltas_and_floors_match_records():
    for label, _, delta, floor in figures.EXECUTED + [figures.APPS]:
        rec_delta, rec_floor, _ = derive.setting_delta_floor_p(FIG_TO_SETTING[label])
        if label == "b2w sandbagging":
            continue  # drawn as the boost (activation minus text AUROC); see test below
        assert abs(delta - rec_delta) <= 0.0015, label
        assert abs(floor - rec_floor) <= 0.0005, label


def test_b2w_sandbagging_is_drawn_as_controlled_delta():
    import json

    rec = json.loads((derive.EVIDENCE / "breadth" / "b2w_result.json").read_text())
    row = rec["sandbagging_wmdp_mmlu"]
    drawn = next(r for r in figures.EXECUTED if r[0] == "b2w sandbagging")
    assert drawn[2] == row["delta"] and drawn[3] == row["floor95"] and drawn[1] == row["text"]


def test_mistral_point_reads_access_fair_record():
    _, x, delta, floor = figures.mistral_decode_cell()
    assert (x, delta, floor) == (0.4915, 0.496, 0.077)
