from ayla.base.pipeline import Pass


def test_pass_options_are_attributes_used_by_repr():
    p = Pass(enabled=True)

    assert p.enabled is True
    assert repr(p) == "Pass(enabled=True)"

    p.enabled = False
    assert repr(p) == "Pass(enabled=False)"
