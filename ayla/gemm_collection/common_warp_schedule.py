"""The Blackwell research schedules use common load/compute/store issuers."""


def validate_common_warp_schedule(configuration):
    """Reject fixed producer/consumer roles, including nested direct schedules.

    Whole-job ping-pong is supported: each participating group loads, computes,
    and writes its own job. Completion waits and collective writeback do not
    assign a separate producer or consumer role.
    """
    if configuration.get("warp_specialize", False):
        raise ValueError("Blackwell schedules forbid warp specialization")
    for attrs in (configuration.get("operand_move_attrs") or {}).values():
        if attrs.get("issuer_warp", 0) != 0 or "participant_warps" in attrs:
            raise ValueError(
                "Blackwell schedules require a shared TMA/MMA/store issuer "
                "without warp specialization; use interleaving, software "
                "pipelining or whole-job ping-pong"
            )
    if configuration.get("operand_issuer_warp", 0) != 0:
        raise ValueError(
            "Blackwell schedules require a shared issuer without warp specialization"
        )
    if "native_schedule" in configuration:
        validate_common_warp_schedule(configuration["native_schedule"])
