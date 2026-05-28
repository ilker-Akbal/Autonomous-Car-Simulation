import time
from typing import Any, Dict, List, Optional


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


SPEED_LIMIT_KMH = {
    "hiz_siniri_20": 20.0,
    "hiz_siniri_30": 30.0,
    "hiz_siniri_40": 40.0,
    "hiz_siniri_50": 50.0,
}

STOP_SIGNS = {
    "dur",
}

SLOW_SIGNS = {
    "yaya_gecidi",
    "okul_gecidi",
    "yol_calismasi",
    "dikkat",
    "yol_ver",
}

INFO_SIGNS = {
    "isikli_isaret_cihazi",
    "iki_yonlu_yol",
    "tunel",
    "park_yeri",
    "park_etmek_yasaktir",
    "ada_etrafinda_donunuz",
}

# Yön / rota kısıtı üreten levhalar.
# Maneuver sözlüğü:
#   left, right, straight, entry
ROUTE_SIGN_RULES = {
    "saga_donulmez": {
        "constraint_type": "turn_forbidden",
        "forbidden_maneuvers": ["right"],
        "description": "Sağa dönüş yasak.",
    },
    "sola_donulmez": {
        "constraint_type": "turn_forbidden",
        "forbidden_maneuvers": ["left"],
        "description": "Sola dönüş yasak.",
    },
    "girisi_olmayan_yol": {
        "constraint_type": "no_entry",
        "forbidden_maneuvers": ["entry", "straight"],
        "description": "Bu yola giriş yasak.",
    },
    "tasit_giremez": {
        "constraint_type": "no_entry",
        "forbidden_maneuvers": ["entry", "straight"],
        "description": "Taşıt giremez.",
    },
    "saga_mecburi_yon": {
        "constraint_type": "mandatory_direction",
        "required_maneuvers": ["right"],
        "description": "Mecburi yön sağ.",
    },
    "sola_mecburi_yon": {
        "constraint_type": "mandatory_direction",
        "required_maneuvers": ["left"],
        "description": "Mecburi yön sol.",
    },
    "ileri_mecburi_yon": {
        "constraint_type": "mandatory_direction",
        "required_maneuvers": ["straight"],
        "description": "Mecburi yön ileri.",
    },
    "ileri_ve_saga_mecburi_yon": {
        "constraint_type": "allowed_direction",
        "allowed_maneuvers": ["straight", "right"],
        "description": "İleri ve sağ yön serbest.",
    },
    "ileri_ve_sola_mecburi_yon": {
        "constraint_type": "allowed_direction",
        "allowed_maneuvers": ["straight", "left"],
        "description": "İleri ve sol yön serbest.",
    },
    "ileriden_saga_mecburi_yon": {
        "constraint_type": "mandatory_direction_ahead",
        "required_maneuvers": ["right"],
        "description": "İleriden sağa mecburi yön.",
    },
    "ileriden_sola_mecburi_yon": {
        "constraint_type": "mandatory_direction_ahead",
        "required_maneuvers": ["left"],
        "description": "İleriden sola mecburi yön.",
    },
    "sagdan_gidiniz": {
        "constraint_type": "lane_side_preference",
        "preferred_side": "right",
        "description": "Sağdan gidiniz.",
    },
    "soldan_gidiniz": {
        "constraint_type": "lane_side_preference",
        "preferred_side": "left",
        "description": "Soldan gidiniz.",
    },
    "serit_duzenleme_levhasi_sag": {
        "constraint_type": "lane_guidance",
        "preferred_side": "right",
        "description": "Şerit düzenleme sağ.",
    },
    "serit_duzenleme_levhasi_sol": {
        "constraint_type": "lane_guidance",
        "preferred_side": "left",
        "description": "Şerit düzenleme sol.",
    },
}


def sign_category(sign_type: Any) -> str:
    sign_type = _norm(sign_type)
    if sign_type in ROUTE_SIGN_RULES:
        return "route_constraint"
    if sign_type in SPEED_LIMIT_KMH or sign_type in STOP_SIGNS or sign_type in SLOW_SIGNS:
        return "decision_event"
    if sign_type in INFO_SIGNS:
        return "info"
    return "unknown"


def _bbox_metrics(det: Dict[str, Any], image_width: Optional[int], image_height: Optional[int]) -> Dict[str, Any]:
    bbox = det.get("bbox") or []
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return {}

    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = w * h

    iw = float(image_width or 0)
    ih = float(image_height or 0)

    out = {
        "bbox_width": round(w, 3),
        "bbox_height": round(h, 3),
        "bbox_area": round(area, 3),
    }

    if iw > 0:
        out["cx_ratio"] = round(cx / iw, 4)
        out["bbox_width_ratio"] = round(w / iw, 4)
    if ih > 0:
        out["cy_ratio"] = round(cy / ih, 4)
        out["bottom_ratio"] = round(y2 / ih, 4)
        out["bbox_height_ratio"] = round(h / ih, 4)
    if iw > 0 and ih > 0:
        out["area_ratio"] = round(area / (iw * ih), 6)

    return out


def _base_sign_payload(det: Dict[str, Any], image_width: Optional[int], image_height: Optional[int]) -> Dict[str, Any]:
    sign_type = _norm(det.get("sign_type", "unknown"))
    yolo_conf = float(det.get("confidence", 0.0) or 0.0)
    sign_conf = float(det.get("sign_confidence", yolo_conf) or 0.0)
    metrics = _bbox_metrics(det, image_width, image_height)

    payload = {
        "stamp": time.time(),
        "source": "perception_sign",
        "ttl_sec": 2.5,
        "sign_type": sign_type,
        "sign_name": det.get("sign_name", sign_type),
        "confidence": round(max(yolo_conf, sign_conf), 4),
        "yolo_confidence": round(yolo_conf, 4),
        "sign_confidence": round(sign_conf, 4),
        "bbox": det.get("bbox"),
        "metrics": metrics,
        "sign_source": det.get("sign_source"),
        "distance_m": det.get("distance_m"),
        "distance_est": det.get("distance_est"),
        "distance_source": det.get("distance_source"),
        "temporal_stable": det.get("temporal_stable"),
        "stable_sign_type": det.get("stable_sign_type"),
        "stable_count": det.get("stable_count"),
        "temporal_vote_total": det.get("temporal_vote_total"),
        "temporal_vote_confidence": det.get("temporal_vote_confidence"),
        "sign_vote_debug": det.get("sign_vote_debug"),
        "frame": "camera_front",
        "position": {
            "frame": "vehicle",
            "distance_m": det.get("distance_m") or det.get("distance_est"),
            "source": det.get("distance_source"),
        },
        "debug": {
            "original_label": det.get("original_label"),
            "source": det.get("source"),
        },
    }

    for key in [
        "cx_ratio",
        "cy_ratio",
        "bottom_ratio",
        "bbox_width_ratio",
        "bbox_height_ratio",
        "area_ratio",
    ]:
        if key in metrics:
            payload[key] = metrics[key]

    return payload


def build_decision_event_from_detection(
    det: Dict[str, Any],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    sign_type = _norm(det.get("sign_type", "unknown"))
    if not sign_type or sign_type == "unknown":
        return None

    payload = _base_sign_payload(det, image_width, image_height)
    payload["category"] = "decision_event"

    if sign_type in SPEED_LIMIT_KMH:
        payload.update({
            "event_type": "speed_limit",
            "speed_limit_kmh": SPEED_LIMIT_KMH[sign_type],
            "target_behavior": "set_speed_limit",
        })
        return payload

    if sign_type in STOP_SIGNS:
        payload.update({
            "event_type": "stop_sign",
            "target_behavior": "stop_then_continue",
        })
        return payload

    if sign_type in SLOW_SIGNS:
        payload.update({
            "event_type": "slow_sign",
            "target_behavior": "slow_or_prepare_to_stop",
        })
        return payload

    if sign_type in INFO_SIGNS:
        payload.update({
            "event_type": "info_sign",
            "target_behavior": "no_direct_decision",
        })
        return payload

    return None


def build_route_constraint_from_detection(
    det: Dict[str, Any],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    sign_type = _norm(det.get("sign_type", "unknown"))
    rule = ROUTE_SIGN_RULES.get(sign_type)
    if not rule:
        return None

    try:
        yolo_conf = float(det.get("confidence", 0.0) or 0.0)
    except Exception:
        yolo_conf = 0.0

    try:
        sign_conf = float(
            det.get("sign_confidence")
            or det.get("traffic_sign_confidence")
            or det.get("classifier_confidence")
            or yolo_conf
            or 0.0
        )
    except Exception:
        sign_conf = 0.0

    sign_source = _norm(det.get("sign_source") or det.get("source"))
    if "classifier_low_conf" in sign_source and max(yolo_conf, sign_conf) < 0.35:
        return None

    if det.get("temporal_stable") is False:
        return None
    try:
        stable_count = int(det.get("stable_count", 2) or 0)
    except Exception:
        stable_count = 0
    if "stable_count" in det and stable_count < 2:
        return None

    if yolo_conf < 0.12 or sign_conf < 0.25:
        return None

    metrics = _bbox_metrics(det, image_width, image_height)
    area_ratio = float(metrics.get("area_ratio", 0.0) or 0.0)
    bottom_ratio = float(metrics.get("bottom_ratio", 0.0) or 0.0)
    h_ratio = float(metrics.get("bbox_height_ratio", 0.0) or 0.0)
    w_ratio = float(metrics.get("bbox_width_ratio", 0.0) or 0.0)
    cx_ratio = float(metrics.get("cx_ratio", 0.5) or 0.5)

    # Rota kısıtları sadece yakındaki, görüntüde anlamlı büyüklükteki levhadan gelsin.
    # Uzak/yan/minik tek-frame levhalar route_intent'i kirletiyordu.
    if area_ratio < 0.00020:
        return None
    if bottom_ratio < 0.16:
        return None
    if h_ratio < 0.018 or w_ratio < 0.010:
        return None
    if cx_ratio < 0.12 or cx_ratio > 0.98:
        return None

    distance_m = det.get("distance_m") or det.get("distance_est")
    try:
        distance_m = float(distance_m) if distance_m is not None else None
    except Exception:
        distance_m = None

    if distance_m is not None and distance_m > 35.0:
        return None

    payload = _base_sign_payload(det, image_width, image_height)
    payload["category"] = "route_constraint"
    payload["route_constraint_filter"] = "accepted_near_stable_candidate"
    payload.update(rule)

    payload.setdefault("forbidden_maneuvers", [])
    payload.setdefault("required_maneuvers", [])
    payload.setdefault("allowed_maneuvers", [])
    payload.setdefault("preferred_side", None)

    return payload


def build_traffic_light_decision_event(tl_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(tl_info, dict):
        return None

    state = _norm(tl_info.get("state", "unknown"))
    if state not in {"red", "yellow", "green"}:
        return None

    payload = {
        "stamp": time.time(),
        "source": "perception_traffic_light",
        "ttl_sec": 1.0,
        "category": "decision_event",
        "event_type": "traffic_light",
        "traffic_light_state": state,
        "target_behavior": {
            "red": "stop",
            "yellow": "slow_or_stop",
            "green": "go",
        }.get(state, "unknown"),
        "confidence": tl_info.get("state_confidence"),
        "det_confidence": tl_info.get("confidence"),
        "bbox": tl_info.get("bbox"),
        "state_source": tl_info.get("state_source"),
        "reason": tl_info.get("color_reason"),
        "route_stopline_relation_required": True,
        "release_must_be_confirmed_by_route_stopline": True,
        "frame": "camera_front",
        "position": {
            "frame": "vehicle",
            "distance_m": tl_info.get("distance_m") or tl_info.get("distance_est"),
            "source": tl_info.get("distance_source"),
        },
        "debug": {
            "active_score": tl_info.get("score"),
            "candidate_count": tl_info.get("candidate_count"),
            "rejected_count": tl_info.get("rejected_count"),
        },
    }

    for key in [
        "distance_m",
        "distance_est",
        "distance_source",
        "tl_cx_ratio",
        "tl_cy_ratio",
        "tl_width_ratio",
        "tl_height_ratio",
        "tl_bottom_ratio",
        "tl_area_ratio",
        "tl_area",
    ]:
        if tl_info.get(key) is not None:
            payload[key] = tl_info.get(key)

    return payload


def road_option_to_maneuver(road_option: Any) -> str:
    value = _norm(road_option)
    mapping = {
        "left": "left",
        "right": "right",
        "straight": "straight",
        "lanefollow": "straight",
        "lane_follow": "straight",
        "void": "unknown",
        "change_lane_left": "left",
        "changelaneleft": "left",
        "change_lane_right": "right",
        "changelaneright": "right",
    }
    return mapping.get(value, "unknown")


def constraint_blocks_maneuver(constraint: Dict[str, Any], maneuver: str) -> bool:
    maneuver = _norm(maneuver)
    if maneuver == "unknown":
        return False

    forbidden = [_norm(x) for x in constraint.get("forbidden_maneuvers", [])]
    required = [_norm(x) for x in constraint.get("required_maneuvers", [])]
    allowed = [_norm(x) for x in constraint.get("allowed_maneuvers", [])]

    if maneuver in forbidden:
        return True

    if required and maneuver not in required:
        return True

    if allowed and maneuver not in allowed:
        return True

    return False


def summarize_constraints_for_log(constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in constraints:
        out.append({
            "sign_type": c.get("sign_type"),
            "constraint_type": c.get("constraint_type"),
            "forbidden": c.get("forbidden_maneuvers", []),
            "required": c.get("required_maneuvers", []),
            "allowed": c.get("allowed_maneuvers", []),
            "preferred_side": c.get("preferred_side"),
            "confidence": c.get("confidence"),
        })
    return out
