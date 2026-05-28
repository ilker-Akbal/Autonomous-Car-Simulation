from teknofest_sim.sign_semantics import constraint_blocks_maneuver, summarize_constraints_for_log


class RouteConstraintManager:
    def __init__(self, timeout_s=2.5):
        self.timeout_s = float(timeout_s or 2.5)

    def filter_constraints(self, constraints, now, last_update_time):
        if not constraints:
            return []
        if last_update_time and now - float(last_update_time) > self.timeout_s:
            return []

        out = []
        for constraint in constraints:
            if not isinstance(constraint, dict):
                continue
            try:
                conf = float(constraint.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0
            metrics = constraint.get("metrics") or {}
            try:
                area_ratio = float(constraint.get("area_ratio", metrics.get("area_ratio", 0.0)) or 0.0)
                bottom_ratio = float(constraint.get("bottom_ratio", metrics.get("bottom_ratio", 0.0)) or 0.0)
                h_ratio = float(constraint.get("bbox_height_ratio", metrics.get("bbox_height_ratio", 0.0)) or 0.0)
            except Exception:
                area_ratio, bottom_ratio, h_ratio = 0.0, 0.0, 0.0

            if conf < 0.25:
                continue
            if area_ratio < 0.00012 and h_ratio < 0.014:
                continue
            if bottom_ratio < 0.12:
                continue

            try:
                stable_count = int(constraint.get("stable_count", 2) or 2)
            except Exception:
                stable_count = 2
            if constraint.get("temporal_stable") is False:
                continue
            if stable_count < 2:
                continue

            out.append(constraint)
        return out

    def allowed_maneuvers(self, maneuver, constraints):
        universe = {"left", "right", "straight", "lane_keep"}
        allowed = set(universe)
        forbidden = set()
        preferred_side = "right"
        for constraint in constraints:
            for item in constraint.get("forbidden_maneuvers", []) or []:
                item = str(item)
                forbidden.add(item)
                allowed.discard(item)

            required = {str(x) for x in constraint.get("required_maneuvers", []) or []}
            if required:
                allowed &= required

            explicit_allowed = {str(x) for x in constraint.get("allowed_maneuvers", []) or []}
            if explicit_allowed:
                allowed &= explicit_allowed

            side = constraint.get("preferred_side")
            if side in {"right", "left"}:
                preferred_side = side

        if maneuver == "straight":
            allowed.add("lane_keep")
        return sorted(allowed), sorted(forbidden), preferred_side

    def evaluate(self, maneuver, constraints):
        blocking = []
        non_blocking = []
        for constraint in constraints:
            if constraint_blocks_maneuver(constraint, maneuver):
                blocking.append(constraint)
            else:
                non_blocking.append(constraint)

        allowed, forbidden, preferred_side = self.allowed_maneuvers(maneuver, constraints)
        if blocking:
            signs = ",".join(str(x.get("sign_type")) for x in blocking[:3])
            return {
                "status": "blocked",
                "reason": f"ROUTE_CONSTRAINT_BLOCKED:route_constraint_blocks_{maneuver}:{signs}|ROUTE_REROUTE_REQUEST|ROUTE_REROUTE_FAILED",
                "allowed_maneuvers": allowed,
                "forbidden_maneuvers": forbidden,
                "preferred_side": preferred_side,
                "blocking_constraints": summarize_constraints_for_log(blocking),
                "non_blocking_constraints": summarize_constraints_for_log(non_blocking),
            }

        if constraints:
            return {
                "status": "constrained",
                "reason": "ROUTE_CONSTRAINT_APPLIED:active_route_constraint",
                "allowed_maneuvers": allowed,
                "forbidden_maneuvers": forbidden,
                "preferred_side": preferred_side,
                "blocking_constraints": [],
                "non_blocking_constraints": summarize_constraints_for_log(non_blocking),
            }

        return {
            "status": "clear",
            "reason": "no_active_route_constraint",
            "allowed_maneuvers": allowed,
            "forbidden_maneuvers": forbidden,
            "preferred_side": preferred_side,
            "blocking_constraints": [],
            "non_blocking_constraints": [],
        }
