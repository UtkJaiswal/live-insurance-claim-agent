from app.live.tools import LiveState, run_tool


def test_locale_needs_a_spoken_answer():
    s = LiveState()
    res, ev = run_tool(s, "set_locale", {"country": "US", "currency": "USD"}, 3.0, None)
    assert res["ok"] is False and ev is None and s.country == ""
    s.last_user_speech_t = 5.0
    res, _ = run_tool(s, "set_locale", {"country": "IN", "currency": "INR"}, 6.0, None)
    assert res["ok"] is True and s.country == "IN"
    # an answer from long ago does not count
    res, _ = run_tool(s, "set_locale", {"country": "GB", "currency": "GBP"}, 60.0, None)
    assert res["ok"] is False and s.country == "IN"


def test_sweep_lifecycle_and_region_tagging():
    s = LiveState(last_user_speech_t=0.5)
    run_tool(s, "set_locale", {"country": "in", "currency": "inr"}, 1.0, None)
    assert (s.country, s.currency) == ("IN", "INR")
    assert not s.sweeping
    run_tool(s, "begin_sweep", {}, 5.0, None)
    assert s.sweeping
    run_tool(s, "mark_region", {"label": "Unit A shelf 1", "kind": "shelf"}, 6.0, "frames/f_000003.jpg")
    _, ev = run_tool(s, "log_item", {"category": "lighting", "description": "brass floor lamp"}, 7.0, "frames/f_000004.jpg")
    assert ev["region"] == "Unit A shelf 1" and ev["frame_ref"] == "frames/f_000004.jpg"
    run_tool(s, "end_sweep", {}, 90.0, None)
    run_tool(s, "end_sweep", {}, 95.0, None)  # second call does not move the end
    assert s.sweep_ended_t == 90.0 and not s.sweeping


def test_unknown_category_falls_back_to_other():
    s = LiveState()
    _, ev = run_tool(s, "log_item", {"category": "spaceship", "description": "x"}, 1.0, None)
    assert ev["category"] == "other"


def test_unknown_tool_is_rejected():
    res, ev = run_tool(LiveState(), "price_everything", {}, 1.0, None)
    assert res["ok"] is False and ev is None
