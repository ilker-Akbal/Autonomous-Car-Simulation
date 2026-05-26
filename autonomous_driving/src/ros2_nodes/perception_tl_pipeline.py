import os
import time


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


class TrafficLightPipeline:
    """
    Sade trafik ışığı pipeline.

    Görevi:
      - traffic_light bbox geldiğinde state üretmek
      - aktif ışığı işaretlemek
      - karar vermemek
      - rota/kontrol işine karışmamak
    """

    KNOWN = {"red", "yellow", "green"}

    def __init__(self, node):
        self.node = node
        self.last_active = None
        self.last_active_time = 0.0

    def norm_state(self, value):
        s = str(value or "unknown").strip().lower()
        aliases = {
            "r": "red",
            "red": "red",
            "kirmizi": "red",
            "kırmızı": "red",
            "y": "yellow",
            "yellow": "yellow",
            "sari": "yellow",
            "sarı": "yellow",
            "g": "green",
            "green": "green",
            "yesil": "green",
            "yeşil": "green",
        }
        return aliases.get(s, "unknown")

    def as_float(self, value, default=0.0):
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def bbox_geom(self, det, frame_w, frame_h):
        bbox = det.get("bbox") or [0, 0, 0, 0]
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except Exception:
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        area = bw * bh
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        return {
            "bbox": [x1, y1, x2, y2],
            "bw": bw,
            "bh": bh,
            "area": area,
            "cx": cx,
            "cy": cy,
            "cx_ratio": cx / max(1.0, float(frame_w)),
            "cy_ratio": cy / max(1.0, float(frame_h)),
        }

    def classify_with_hsv(self, frame, bbox):
        fn = getattr(self.node, "classify_traffic_light_hsv_strict", None)
        if fn is None:
            return "unknown", 0.0, {}, "hsv_not_available"

        try:
            state, scores, reason = fn(frame, bbox)
            state = self.norm_state(state)
            if state in self.KNOWN:
                return state, 0.55, scores or {}, reason
            return "unknown", 0.0, scores or {}, reason
        except Exception as exc:
            return "unknown", 0.0, {}, f"hsv_error:{exc}"

    def classify_with_model(self, frame, bbox):
        fn = getattr(self.node, "classify_traffic_light_state_model", None)
        if fn is None:
            return "unknown", 0.0, {}, "classifier_not_available"

        try:
            result = fn(frame, bbox)
        except Exception as exc:
            return "unknown", 0.0, {}, f"classifier_error:{exc}"

        if not isinstance(result, dict):
            return "unknown", 0.0, {}, "classifier_none"

        state = self.norm_state(result.get("state", "unknown"))
        conf = self.as_float(result.get("confidence"), 0.0)
        probs = result.get("probs") or {}
        reason = str(result.get("reason", "-"))
        return state, conf, probs, reason

    def best_prob(self, probs):
        if not isinstance(probs, dict):
            return "unknown", 0.0

        best_state = "unknown"
        best_conf = 0.0

        for st in ("red", "yellow", "green"):
            c = self.as_float(probs.get(st), 0.0)
            if c > best_conf:
                best_state = st
                best_conf = c

        return best_state, best_conf

    def classify_candidate(self, frame, det):
        out = dict(det)
        bbox = out.get("bbox") or [0, 0, 0, 0]

        hsv_state, hsv_conf, hsv_scores, hsv_reason = self.classify_with_hsv(frame, bbox)
        cls_state, cls_conf, cls_probs, cls_reason = self.classify_with_model(frame, bbox)
        best_state, best_conf = self.best_prob(cls_probs)

        red_thr = _env_float("TL_SIMPLE_RED_CONF", 0.40)
        green_thr = _env_float("TL_SIMPLE_GREEN_CONF", 0.40)
        yellow_thr = _env_float("TL_SIMPLE_YELLOW_CONF", 0.55)

        state = "unknown"
        conf = 0.0
        source = "unknown"
        reason = "no_known_state"

        if cls_state == "red" and cls_conf >= red_thr:
            state, conf, source, reason = "red", cls_conf, "classifier", cls_reason
        elif cls_state == "green" and cls_conf >= green_thr:
            state, conf, source, reason = "green", cls_conf, "classifier", cls_reason
        elif cls_state == "yellow" and cls_conf >= yellow_thr:
            state, conf, source, reason = "yellow", cls_conf, "classifier", cls_reason
        elif hsv_state in self.KNOWN:
            state, conf, source, reason = hsv_state, hsv_conf, "hsv_fallback", hsv_reason
        elif best_state == "red" and best_conf >= red_thr:
            state, conf, source, reason = "red", best_conf, "classifier_probs", cls_reason
        elif best_state == "green" and best_conf >= green_thr:
            state, conf, source, reason = "green", best_conf, "classifier_probs", cls_reason
        elif best_state == "yellow" and best_conf >= yellow_thr:
            state, conf, source, reason = "yellow", best_conf, "classifier_probs", cls_reason

        out["traffic_light_state"] = state
        out["traffic_light_state_confidence"] = float(conf)
        out["tl_state_confidence"] = float(conf)
        out["state_confidence"] = float(conf)
        out["state_conf"] = float(conf)
        out["traffic_light_state_source"] = source
        out["state_source"] = source
        out["traffic_light_state_probs"] = cls_probs
        out["traffic_light_hsv_state"] = hsv_state
        out["traffic_light_color_scores"] = hsv_scores
        out["traffic_light_color_reason"] = reason
        out["traffic_light_classifier_reason"] = cls_reason
        return out

    def select_active(self, candidates, frame_w, frame_h):
        usable = []

        for det in candidates:
            geom = self.bbox_geom(det, frame_w, frame_h)
            det["tl_cx_ratio"] = round(geom["cx_ratio"], 4)
            det["tl_cy_ratio"] = round(geom["cy_ratio"], 4)
            det["tl_area"] = round(geom["area"], 2)

            state = self.norm_state(det.get("traffic_light_state"))
            conf = self.as_float(det.get("state_confidence"), 0.0)
            det_conf = self.as_float(det.get("confidence"), 0.0)

            if state not in self.KNOWN:
                det["tl_pipeline_reject_reason"] = "unknown_state"
                continue

            if det_conf < _env_float("TL_SIMPLE_MIN_DET_CONF", 0.06):
                det["tl_pipeline_reject_reason"] = "low_det_conf"
                continue

            if conf < _env_float("TL_SIMPLE_ACTIVE_STATE_CONF", 0.30):
                det["tl_pipeline_reject_reason"] = "low_state_conf"
                continue

            target_x = _env_float("TL_ACTIVE_TARGET_X_RATIO", 0.52)
            target_y = _env_float("TL_ACTIVE_TARGET_Y_RATIO", 0.24)

            x_score = 1.0 - min(1.0, abs(geom["cx_ratio"] - target_x) / 0.55)
            y_score = 1.0 - min(1.0, abs(geom["cy_ratio"] - target_y) / 0.55)
            size_score = min(1.0, geom["area"] / _env_float("TL_ACTIVE_SIZE_NORM_AREA", 500.0))

            score = (
                0.35 * x_score
                + 0.25 * y_score
                + 0.20 * size_score
                + 0.20 * conf
            )

            det["tl_active_score"] = round(score, 4)
            usable.append(det)

        if not usable:
            return None, candidates

        usable.sort(key=lambda d: d.get("tl_active_score", 0.0), reverse=True)
        return usable[0], [d for d in candidates if d not in usable]

    def process(self, frame, detections, frame_w, frame_h):
        out = []
        tl_candidates = []

        for det in detections:
            if det.get("label") != "traffic_light":
                out.append(det)
                continue

            enriched = self.classify_candidate(frame, det)
            enriched["active_traffic_light"] = False
            tl_candidates.append(enriched)

        active, rejected = self.select_active(tl_candidates, frame_w, frame_h)

        info = {
            "state": "unknown",
            "confidence": None,
            "state_confidence": None,
            "state_source": None,
            "state_probs": None,
            "color_scores": None,
            "color_reason": None,
            "bbox": None,
            "candidate_count": len(tl_candidates),
            "rejected_count": len(rejected),
            "score": None,
            "roi": None,
        }

        if active is not None:
            active["active_traffic_light"] = True
            self.last_active = dict(active)
            self.last_active_time = time.time()

            info = {
                "state": active.get("traffic_light_state", "unknown"),
                "confidence": float(active.get("confidence", 0.0)),
                "state_confidence": self.as_float(active.get("state_confidence"), 0.0),
                "state_source": active.get("traffic_light_state_source"),
                "state_probs": active.get("traffic_light_state_probs"),
                "color_scores": active.get("traffic_light_color_scores"),
                "color_reason": active.get("traffic_light_color_reason"),
                "bbox": active.get("bbox"),
                "candidate_count": len(tl_candidates),
                "rejected_count": len(rejected),
                "score": active.get("tl_active_score"),
                "roi": None,
            }

        out.extend(tl_candidates)
        return out, info, tl_candidates, rejected
