def normalize_decision_command(command):
    """Keep the public decision command internally consistent."""
    out = dict(command or {})
    decision = str(out.get("decision", "GO") or "GO").upper()
    should_stop = bool(out.get("should_stop", False))
    emergency = bool(out.get("emergency_brake", False))
    reason = str(out.get("reason", "") or "")

    must_stop = should_stop or emergency or decision == "STOP"
    if must_stop:
        out["decision"] = "STOP"
        out["target_speed"] = 0.0
        out["target_speed_mps"] = 0.0
        out["target_speed_kmh"] = 0.0
        out["should_stop"] = True
        out["stop_required"] = True
        out["risk"] = "HIGH" if emergency else out.get("risk", "MEDIUM_STOP")
        out["stop_reason"] = out.get("stop_reason") or reason or "normalized_stop"
        if "DECISION_NORMALIZED_STOP" not in reason:
            out["reason"] = f"{reason}|DECISION_NORMALIZED_STOP" if reason else "DECISION_NORMALIZED_STOP"
        return out

    if decision not in {"GO", "SLOW"}:
        decision = "GO"
        out["decision"] = decision

    out["should_stop"] = False
    out["stop_required"] = False
    out["emergency_brake"] = False
    out["stop_reason"] = None
    try:
        speed = float(out.get("target_speed_mps", out.get("target_speed", 0.0)) or 0.0)
    except Exception:
        speed = 0.0

    if decision == "GO":
        speed = max(0.0, speed)
    else:
        speed = max(0.0, speed)

    out["target_speed"] = speed
    out["target_speed_mps"] = speed
    return out


def route_blocked_stop(route_intent):
    if not isinstance(route_intent, dict):
        return False, "route_intent_missing"
    status = str(route_intent.get("route_decision_status", "clear") or "clear").lower()
    if status == "blocked":
        return True, str(route_intent.get("route_decision_reason", "ROUTE_CONSTRAINT_BLOCKED"))
    return False, status
