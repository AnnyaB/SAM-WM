from coolworld.planner import CandidateOutcome, choose_cooling_action


def test_planner_requires_interval_to_remain_cooling():
    result = choose_cooling_action(
        [
            CandidateOutcome("uncertain", -2.0, -4.0, 0.4, 0.8, 100.0),
            CandidateOutcome("supported", -1.0, -1.4, -0.2, 0.8, 120.0),
        ]
    )
    assert result.selected_id == "supported"


def test_planner_rejects_low_support():
    result = choose_cooling_action([CandidateOutcome("x", -2, -3, -1, 0.01, 1)])
    assert result.status == "NO_SUPPORTED_COOLING_ACTION"
