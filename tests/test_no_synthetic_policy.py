from coolworld.evidence import EvidenceKind


def test_evidence_enum_contains_no_simulator_or_synthetic_kind() -> None:
    names = {item.name.lower() for item in EvidenceKind}
    values = {item.value.lower() for item in EvidenceKind}
    forbidden = {"simulator", "synthetic", "fake", "demo", "smoke"}
    assert not any(any(word in value for word in forbidden) for value in names | values)
