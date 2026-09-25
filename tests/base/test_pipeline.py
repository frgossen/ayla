from ayla.base.pipeline import Pass


def test_pass_options_are_attributes_used_by_repr():
    p = Pass(enabled=True)

    assert getattr(p, "enabled") is True
    assert repr(p) == "Pass(enabled=True)"

    setattr(p, "enabled", False)
    assert repr(p) == "Pass(enabled=False)"
