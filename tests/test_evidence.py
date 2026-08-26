from coolworld.evidence import digest_json


def test_digest_is_order_invariant():
    assert digest_json({"a": 1, "b": 2}) == digest_json({"b": 2, "a": 1})
