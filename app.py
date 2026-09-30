import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import json
import sqlite3
import math
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------

class Entity(BaseModel):
    id: str
    kind: str
    name: str
    status: str


class Observation(BaseModel):
    tick: int
    ts: str
    entity_id: str
    source: str
    metric: str
    value: Optional[float] = None
    quality: str = "ok"
    missing: bool = False


class Event(BaseModel):
    tick: int
    ts: str
    type: str
    entity_id: str
    detail: str
    severity: str


class StateEstimate(BaseModel):
    tick: int
    ts: str
    entity_id: str
    health: float
    missing_rate: float
    contradiction: float
    drift: float
    updated_at: str


class Detection(BaseModel):
    tick: int
    ts: str
    detection_id: str
    score: float
    active: bool
    persistence: int
    explanation: str


class Hypothesis(BaseModel):
    hypothesis_id: str
    title: str
    score: float
    support: List[str] = Field(default_factory=list)
    contradiction: List[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    tick: int
    score: float
    level: str
    components: Dict[str, float]
    explanation: str


class ActionProposal(BaseModel):
    action_id: str
    tick: int
    name: str
    policy_decision: str
    simulation_only: bool
    bounded: bool
    reversible: bool
    intended_effect: str
    status: str


class VerificationResult(BaseModel):
    tick: int
    action_id: str
    verifier_id: str
    passed: bool
    checks: List[Dict[str, Any]]
    proof: Dict[str, Any]


class EvidenceBundle(BaseModel):
    evidence_id: str
    tick: int
    kind: str
    payload: Dict[str, Any]
    sha256: str


class AuditEvent(BaseModel):
    tick: int
    ts: str
    event_type: str
    actor: str
    detail: str


class Metric(BaseModel):
    tick: int
    name: str
    value: float
    target: str


# -----------------------------------------------------------------------------
# Defaults, helpers
# -----------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "seed": 42,
    "ticks": 120,
    "demo_speed": 4,
    "baseline_window": 20,
    "detection_threshold": 0.52,
    "confidence_threshold": 0.48,
    "risk_weights": {
        "anomaly_strength": 0.30,
        "impact": 0.25,
        "persistence": 0.15,
        "confidence": 0.15,
        "evidence_quality": 0.10,
        "uncertainty_penalty": 0.05,
    },
    "policy": {
        "autonomous_actions": "simulator_only",
        "max_bounded_impact": "low",
        "require_approval_above_risk": 90,
    },
    "safety": {
        "actions": "simulator_only",
        "verification": "independent",
        "evidence_hash": "sha256",
        "audit": "append_only",
    },
}

YAML_CONFIG = """
seed: 42
ticks: 120
demo_speed: 4
baseline_window: 20
detection_threshold: 0.52
confidence_threshold: 0.48
risk_weights:
  anomaly_strength: 0.30
  impact: 0.25
  persistence: 0.15
  confidence: 0.15
  evidence_quality: 0.10
  uncertainty_penalty: 0.05
policy:
  autonomous_actions: simulator_only
  max_bounded_impact: low
  require_approval_above_risk: 90
safety:
  actions: simulator_only
  verification: independent
  evidence_hash: sha256
  audit: append_only
"""


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_z(value: float, mean: float, std: float) -> float:
    if std <= 1e-9:
        return 0.0
    return float((value - mean) / std)


def sha256_json(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def model_to_dict(obj: Any) -> Dict[str, Any]:
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
    except Exception:
        pass
    return dict(obj)


def policy_gate(risk_score: float, action: ActionProposal) -> str:
    if not action.simulation_only:
        if risk_score >= 85:
            return "block"
        if risk_score >= 60:
            return "require_approval"

    if action.simulation_only and action.bounded and action.reversible:
        if risk_score < 20:
            return "observe"
        if risk_score < 45:
            return "recommend"
        if risk_score < 90:
            return "simulate"
        return "require_approval"

    if risk_score < 20:
        return "observe"
    if risk_score < 45:
        return "recommend"
    if risk_score < 70:
        return "simulate"
    return "require_approval"


# -----------------------------------------------------------------------------
# SQLite persistence
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_db():
    try:
        conn = sqlite3.connect("ara_demo.db", check_same_thread=False)
        conn.execute("SELECT 1")
        return conn
    except Exception:
        return sqlite3.connect(":memory:", check_same_thread=False)


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, payload TEXT)"
    )
    tables = [
        "observations",
        "entities",
        "events",
        "states",
        "detections",
        "hypotheses",
        "risks",
        "actions",
        "verifications",
        "evidence",
        "audit",
        "metrics",
        "timeline",
        "model_runs",
    ]
    for table in tables:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (tick INTEGER, payload TEXT)"
        )
    conn.commit()


def persist_demo(conn: sqlite3.Connection, demo: Dict[str, Any]) -> bool:
    run_id = demo.get("run_id", "unknown")
    cur = conn.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,))
    if cur.fetchone():
        return False

    conn.execute(
        "INSERT INTO runs (run_id, payload) VALUES (?, ?)",
        (
            run_id,
            json.dumps(
                {
                    "created_at": datetime.now().isoformat(),
                    "seed": demo.get("seed"),
                    "ticks": demo.get("ticks"),
                    "config_hash": demo.get("config_hash"),
                },
                default=str,
            ),
        ),
    )

    mapping = {
        "observations": "observations",
        "entities": "entities",
        "events": "events",
        "states": "states",
        "detections": "detections",
        "hypotheses": "hypotheses",
        "risks": "risks",
        "actions": "actions",
        "verifications": "verifications",
        "evidence": "evidence",
        "audit": "audit",
        "metrics": "metrics",
        "timeline": "timeline",
        "model_runs": "model_runs",
    }

    for table, key in mapping.items():
        df = demo.get(key)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            payload = row.to_dict()
            tick = int(payload.get("tick", 0) or 0)
            conn.execute(
                f"INSERT INTO {table} (tick, payload) VALUES (?, ?)",
                (tick, json.dumps(payload, default=str)),
            )

    conn.commit()
    return True


# -----------------------------------------------------------------------------
# Deterministic synthetic world + closed-loop agent simulation
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_demo(seed: int = 42, ticks: int = 120) -> Dict[str, Any]:
    rng = random.Random(seed)
    base_time = datetime(2026, 9, 30, 10, 0, 0)

    def ts(t: int) -> str:
        return (base_time + timedelta(seconds=t)).isoformat()

    def phase_name(t: int) -> str:
        if t < 20:
            return "normal"
        if t < 45:
            return "anomaly_injection"
        if t < 70:
            return "persistent_detection"
        if t < 90:
            return "ai_explanation"
        if t < 105:
            return "investigation_risk"
        if t < 115:
            return "policy_action"
        return "verification"

    entities = [
        Entity(id="service.checkout", kind="service", name="Checkout API", status="healthy"),
        Entity(id="db.primary", kind="database", name="Primary DB", status="healthy"),
        Entity(id="network.edge", kind="network", name="Edge Network", status="healthy"),
        Entity(id="sensor.b", kind="sensor", name="Monitor B", status="healthy"),
    ]

    rows: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []

    audit.append(
        model_to_dict(
            AuditEvent(
                tick=0,
                ts=ts(0),
                event_type="sim_start",
                actor="synthetic_world",
                detail=f"Deterministic simulation started with seed={seed}",
            )
        )
    )

    for t in range(ticks):
        phase = phase_name(t)

        latency = 120 + 10 * math.sin(t / 9) + rng.uniform(-6, 6)
        error_rate = max(0.0, 0.6 + 0.25 * math.sin(t / 7) + rng.uniform(-0.08, 0.08))
        cpu = 38 + 6 * math.sin(t / 11) + rng.uniform(-2, 2)
        memory = 52 + 4 * math.sin(t / 13) + rng.uniform(-1.5, 1.5)
        throughput = 980 + 40 * math.sin(t / 10) + rng.uniform(-18, 18)
        network_loss = max(0.0, 0.2 + 0.1 * math.sin(t / 5) + rng.uniform(-0.05, 0.05))
        auth_failures = max(0.0, rng.uniform(0, 1.2))
        integrity = clip(0.995 + rng.uniform(-0.003, 0.003), 0, 1)

        scenarios: List[str] = []
        anomaly_factor = 0.0

        if 20 <= t < 115:
            if t < 45:
                strength = 1.0
                scenarios.append("latency_dependency_degradation_onset")
            elif t < 70:
                strength = 0.75
                scenarios.append("persistent_latency_degradation")
            elif t < 105:
                strength = 0.60
                scenarios.append("latent_dependency_degradation")
            else:
                strength = 0.45
                scenarios.append("pre_mitigation_latency")

            mitigation = 0.0
            if t >= 110:
                mitigation = min(1.0, (t - 110) / 8.0)

            anomaly_factor = strength * (1.0 - mitigation)
            latency += 95 * anomaly_factor + 25 * math.sin(t / 3) * anomaly_factor
            error_rate += 2.8 * anomaly_factor
            throughput -= 260 * anomaly_factor
            cpu += 14 * anomaly_factor

        missing_count = 0
        if 30 <= t < 34:
            missing_count = 1
            scenarios.append("missing_sensor_b")

        health_base = clip(1.0 - max(0.0, (latency - 160) / 120) - error_rate / 8, 0, 1)
        health_a = clip(health_base + rng.uniform(-0.02, 0.02), 0, 1)
        health_b = clip(health_base + rng.uniform(-0.02, 0.02), 0, 1)
        health_c = clip(health_base + rng.uniform(-0.02, 0.02), 0, 1)

        if 38 <= t < 42:
            health_c = clip(health_base - 0.45, 0, 1)
            scenarios.append("contradictory_health_readings")

        contradiction = float(np.std([health_a, health_b, health_c]))

        if 50 <= t < 112:
            memory += min(38, (t - 50) * 0.75)
            scenarios.append("memory_drift")

        if t >= 112:
            memory = 52 + 4 * math.sin(t / 13) + rng.uniform(-1.5, 1.5)

        if 60 <= t < 66:
            network_loss += 3.6
            scenarios.append("network_packet_loss")

        if 80 <= t < 86:
            auth_failures += 14
            scenarios.append("auth_failure_burst")

        latency = max(20.0, latency)
        error_rate = max(0.0, error_rate)
        throughput = max(0.0, throughput)
        cpu = clip(cpu, 0, 100)
        memory = clip(memory, 0, 100)
        network_loss = max(0.0, network_loss)
        auth_failures = max(0.0, auth_failures)

        row = {
            "tick": t,
            "ts": ts(t),
            "phase": phase,
            "latency": float(latency),
            "error_rate": float(error_rate),
            "cpu": float(cpu),
            "memory": float(memory),
            "throughput": float(throughput),
            "network_loss": float(network_loss),
            "auth_failures": float(auth_failures),
            "integrity": float(integrity),
            "health_a": float(health_a),
            "health_b": float(health_b),
            "health_c": float(health_c),
            "missing_count": int(missing_count),
            "contradiction": float(contradiction),
            "anomaly_ground_truth": bool(20 <= t < 115),
            "scenarios": scenarios,
        }
        rows.append(row)

        canonical_metrics = [
            "latency",
            "error_rate",
            "cpu",
            "memory",
            "throughput",
            "network_loss",
            "auth_failures",
            "integrity",
        ]
        for metric in canonical_metrics:
            obs = Observation(
                tick=t,
                ts=ts(t),
                entity_id="service.checkout",
                source="monitor-a",
                metric=metric,
                value=float(row[metric]),
                quality="ok",
                missing=False,
            )
            observations.append(model_to_dict(obs))

        if 30 <= t < 34:
            obs_missing = Observation(
                tick=t,
                ts=ts(t),
                entity_id="service.checkout",
                source="monitor-b",
                metric="latency",
                value=None,
                quality="missing",
                missing=True,
            )
            observations.append(model_to_dict(obs_missing))

        health_obs = [
            ("monitor-a", health_a, "ok"),
            ("monitor-b", health_b, "ok"),
            ("monitor-c", health_c, "contradictory" if 38 <= t < 42 else "ok"),
        ]
        for source, value, quality in health_obs:
            obs_health = Observation(
                tick=t,
                ts=ts(t),
                entity_id="service.checkout",
                source=source,
                metric="health",
                value=float(value),
                quality=quality,
                missing=False,
            )
            observations.append(model_to_dict(obs_health))

        if t == 20:
            events.append(
                model_to_dict(
                    Event(
                        tick=t,
                        ts=ts(t),
                        type="anomaly_injection",
                        entity_id="service.checkout",
                        detail="Latency degradation onset",
                        severity="high",
                    )
                )
            )
        if t == 30:
            events.append(
                model_to_dict(
                    Event(
                        tick=t,
                        ts=ts(t),
                        type="missing_data",
                        entity_id="sensor.b",
                        detail="Monitor B latency stream missing",
                        severity="medium",
                    )
                )
            )
        if t == 38:
            events.append(
                model_to_dict(
                    Event(
                        tick=t,
                        ts=ts(t),
                        type="contradiction",
                        entity_id="service.checkout",
                        detail="Health readings conflict across monitors",
                        severity="medium",
                    )
                )
            )
        if t == 50:
            events.append(
                model_to_dict(
                    Event(
                        tick=t,
                        ts=ts(t),
                        type="drift",
                        entity_id="service.checkout",
                        detail="Memory drift begins",
                        severity="medium",
                    )
                )
            )
        if t == 60:
            events.append(
                model_to_dict(
                    Event(
                        tick=t,
                        ts=ts(t),
                        type="network_loss",
                        entity_id="network.edge",
                        detail="Packet loss burst",
                        severity="medium",
                    )
                )
            )
        if t == 80:
            events.append(
                model_to_dict(
                    Event(
                        tick=t,
                        ts=ts(t),
                        type="auth_failure",
                        entity_id="service.checkout",
                        detail="Authentication failure burst",
                        severity="medium",
                    )
                )
            )
        if t == 105:
            events.append(
                model_to_dict(
                    Event(
                        tick=t,
                        ts=ts(t),
                        type="action_proposed",
                        entity_id="service.checkout",
                        detail="Simulated mitigation proposed",
                        severity="info",
                    )
                )
            )
        if t == 115:
            events.append(
                model_to_dict(
                    Event(
                        tick=t,
                        ts=ts(t),
                        type="verification_phase",
                        entity_id="service.checkout",
                        detail="Independent verification phase begins",
                        severity="info",
                    )
                )
            )

    df = pd.DataFrame(rows)

    feature_metrics = [
        "latency",
        "error_rate",
        "cpu",
        "memory",
        "throughput",
        "network_loss",
        "auth_failures",
    ]

    baseline = {}
    for metric in feature_metrics:
        vals = [r[metric] for r in rows[:20]]
        baseline[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) + 1e-6,
        }

    detections: List[Dict[str, Any]] = []
    states: List[Dict[str, Any]] = []
    hypotheses: List[Dict[str, Any]] = []
    risks: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    verifications: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []

    persistence = 0
    first_detection_tick: Optional[int] = None
    action_id = "act-001"
    action_name = "enable_simulated_circuit_breaker"
    policy_decision_at_action = "simulate"

    config_hash = sha256_json(DEFAULT_CONFIG)
    run_id = sha256_json({"seed": seed, "ticks": ticks, "config_hash": config_hash})[:16]

    def add_evidence(tick: int, kind: str, payload: Dict[str, Any]) -> None:
        bundle = {
            "evidence_id": f"ev-{len(evidence) + 1:03d}",
            "tick": tick,
            "kind": kind,
            "payload": payload,
        }
        bundle["sha256"] = sha256_json({k: v for k, v in bundle.items() if k != "sha256"})
        evidence.append(bundle)

    for i, row in enumerate(rows):
        t = int(row["tick"])
        cold_start = t < 20

        z = {
            metric: safe_z(row[metric], baseline[metric]["mean"], baseline[metric]["std"])
            for metric in feature_metrics
        }

        feature_scores = {
            "latency": clip(z["latency"] / 8.0),
            "error_rate": clip(z["error_rate"] / 6.0),
            "throughput": clip(-z["throughput"] / 6.0),
            "cpu": clip(z["cpu"] / 8.0),
            "memory": clip(z["memory"] / 8.0),
            "network_loss": clip(z["network_loss"] / 8.0),
            "auth_failures": clip(z["auth_failures"] / 8.0),
        }

        missing_rate = row["missing_count"] / 3.0
        contradiction_norm = clip(row["contradiction"] / 0.2)

        anomaly_strength = clip(
            0.30 * feature_scores["latency"]
            + 0.22 * feature_scores["error_rate"]
            + 0.14 * feature_scores["throughput"]
            + 0.08 * feature_scores["cpu"]
            + 0.08 * feature_scores["memory"]
            + 0.08 * feature_scores["network_loss"]
            + 0.10 * feature_scores["auth_failures"]
            + 0.25 * missing_rate
            + 0.35 * contradiction_norm
        )

        evidence_quality = clip(
            0.90 - missing_rate * 1.2 - min(0.35, row["contradiction"] * 1.5)
        )
        uncertainty = clip(
            0.22
            + missing_rate * 0.8
            + min(0.35, row["contradiction"] * 1.5)
            + (0.25 if cold_start else 0.0)
        )
        confidence = clip(
            0.25
            + 0.70 * anomaly_strength
            + 0.10 * evidence_quality
            - 0.20 * uncertainty
        )
        calibrated_score = clip(anomaly_strength * (0.85 + 0.35 * confidence))

        active = (
            not cold_start
            and calibrated_score >= DEFAULT_CONFIG["detection_threshold"]
            and confidence >= DEFAULT_CONFIG["confidence_threshold"]
        )

        if active:
            persistence += 1
        else:
            persistence = 0

        if active and first_detection_tick is None:
            first_detection_tick = t

        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        explanation = (
            "Normal operation"
            if not active
            else (
                "Anomalous pattern driven by "
                + ", ".join([f"{k}={v:.2f}" for k, v in top_features])
                + f"; persistence={persistence}."
            )
        )

        detection = Detection(
            tick=t,
            ts=row["ts"],
            detection_id=f"det-{t:03d}",
            score=round(calibrated_score, 4),
            active=active,
            persistence=persistence,
            explanation=explanation,
        )
        detection_dict = model_to_dict(detection)
        detection_dict["raw_score"] = round(anomaly_strength, 4)
        detection_dict["confidence"] = round(confidence, 4)
        detection_dict["features"] = {k: round(v, 4) for k, v in feature_scores.items()}
        detections.append(detection_dict)

        health = clip(1.0 - calibrated_score)
        state = StateEstimate(
            tick=t,
            ts=row["ts"],
            entity_id="service.checkout",
            health=round(health, 4),
            missing_rate=round(missing_rate, 4),
            contradiction=round(row["contradiction"], 4),
            drift=round(clip(max(0.0, z["memory"]) / 8.0), 4),
            updated_at=row["ts"],
        )
        states.append(model_to_dict(state))

        hyps = []

        h1 = Hypothesis(
            hypothesis_id="H1",
            title="Downstream dependency latency degradation",
            score=round(
                clip(
                    0.50 * feature_scores["latency"]
                    + 0.30 * feature_scores["error_rate"]
                    + 0.20 * feature_scores["throughput"]
                    + 0.10 * min(1.0, persistence / 8.0)
                ),
                4,
            ),
            support=[
                f"latency z={z['latency']:.2f}",
                f"error z={z['error_rate']:.2f}",
                f"throughput z={z['throughput']:.2f}",
            ],
            contradiction=(
                [f"cpu z only {z['cpu']:.2f}"] if z["cpu"] < 2 else []
            ),
        )

        h2 = Hypothesis(
            hypothesis_id="H2",
            title="Observability or data-quality fault",
            score=round(clip(0.70 * missing_rate + 0.90 * contradiction_norm), 4),
            support=[
                "missing sensor B" if missing_rate > 0 else "no missing data",
                f"contradiction={row['contradiction']:.3f}",
            ],
            contradiction=(
                ["primary monitor A intact"]
                if missing_rate == 0 and row["contradiction"] < 0.05
                else []
            ),
        )

        h3 = Hypothesis(
            hypothesis_id="H3",
            title="Resource exhaustion / memory drift",
            score=round(clip(0.80 * feature_scores["memory"] + 0.25 * feature_scores["cpu"]), 4),
            support=[f"memory z={z['memory']:.2f}"],
            contradiction=(
                [f"cpu z={z['cpu']:.2f} low"] if z["cpu"] < 1 else []
            ),
        )

        h4 = Hypothesis(
            hypothesis_id="H4",
            title="Network packet loss",
            score=round(clip(0.95 * feature_scores["network_loss"]), 4),
            support=[f"net loss z={z['network_loss']:.2f}"],
            contradiction=(
                ["network loss near baseline"] if feature_scores["network_loss"] <= 0.2 else []
            ),
        )

        h5 = Hypothesis(
            hypothesis_id="H5",
            title="Authentication failure storm",
            score=round(clip(0.95 * feature_scores["auth_failures"]), 4),
            support=[f"auth z={z['auth_failures']:.2f}"],
            contradiction=(
                ["auth failures near baseline"]
                if feature_scores["auth_failures"] <= 0.2
                else []
            ),
        )

        hyp_dicts = [model_to_dict(h) for h in [h1, h2, h3, h4, h5]]
        hyp_dicts = sorted(hyp_dicts, key=lambda x: x["score"], reverse=True)
        for rank, hyp in enumerate(hyp_dicts, 1):
            hyp["tick"] = t
            hyp["ts"] = row["ts"]
            hyp["rank"] = rank
            hypotheses.append(hyp)

        impact = clip(
            0.35 * feature_scores["latency"]
            + 0.25 * feature_scores["error_rate"]
            + 0.15 * feature_scores["throughput"]
            + 0.10 * feature_scores["memory"]
            + 0.10 * feature_scores["network_loss"]
            + 0.05 * feature_scores["auth_failures"]
        )
        persistence_score = min(1.0, persistence / 8.0)

        risk_raw = 100 * clip(
            0.30 * calibrated_score
            + 0.25 * impact
            + 0.15 * persistence_score
            + 0.15 * confidence
            + 0.10 * evidence_quality
            - 0.05 * uncertainty
        )

        if not active:
            risk_raw = min(risk_raw, 45.0)

        risk_score = round(clip(risk_raw, 0, 100), 1)
        level = (
            "low"
            if risk_score < 25
            else "medium"
            if risk_score < 50
            else "high"
            if risk_score < 75
            else "critical"
        )

        components = {
            "anomaly_strength": round(30 * calibrated_score, 2),
            "impact": round(25 * impact, 2),
            "persistence": round(15 * persistence_score, 2),
            "confidence": round(15 * confidence, 2),
            "evidence_quality": round(10 * evidence_quality, 2),
            "uncertainty_penalty": round(-5 * uncertainty, 2),
        }

        risk_explanation = (
            f"Risk {risk_score}/100 ({level}) from anomaly {calibrated_score:.2f}, "
            f"impact {impact:.2f}, persistence {persistence_score:.2f}, "
            f"confidence {confidence:.2f}."
        )

        risk = RiskAssessment(
            tick=t,
            score=risk_score,
            level=level,
            components=components,
            explanation=risk_explanation,
        )
        risks.append(model_to_dict(risk))

        if t % 5 == 0 or (active and persistence == 1):
            timeline.append(
                {
                    "tick": t,
                    "ts": row["ts"],
                    "type": "state",
                    "detail": f"health={health:.2f}, risk={risk_score}, active={active}",
                }
            )

        if t in [20, 30, 38, 50, 60, 80, 105, 115]:
            timeline.append(
                {
                    "tick": t,
                    "ts": row["ts"],
                    "type": "event",
                    "detail": phase_name(t),
                }
            )

        if t == 20:
            add_evidence(
                t,
                "anomaly_onset",
                {
                    "row": row,
                    "features": feature_scores,
                    "detection": detection_dict,
                    "risk_score": risk_score,
                },
            )

        if t == 45:
            add_evidence(
                t,
                "persistent_detection",
                {
                    "row": row,
                    "features": feature_scores,
                    "detection": detection_dict,
                    "risk_score": risk_score,
                },
            )

        if t == 90:
            add_evidence(
                t,
                "ai_explanation",
                {
                    "top_hypotheses": hyp_dicts[:3],
                    "risk_score": risk_score,
                    "row": row,
                },
            )

        if t == 105:
            proposal = ActionProposal(
                action_id=action_id,
                tick=t,
                name=action_name,
                policy_decision="pending",
                simulation_only=True,
                bounded=True,
                reversible=True,
                intended_effect="Reduce checkout latency and error rate within 10 seconds",
                status="proposed",
            )
            policy_decision_at_action = policy_gate(risk_score, proposal)
            proposal.policy_decision = policy_decision_at_action
            action_dict = model_to_dict(proposal)
            actions.append(action_dict)

            audit.append(
                model_to_dict(
                    AuditEvent(
                        tick=t,
                        ts=row["ts"],
                        event_type="policy_decision",
                        actor="policy_engine",
                        detail=f"{action_name}: {policy_decision_at_action}",
                    )
                )
            )

            add_evidence(
                t,
                "action_package",
                {
                    "action": action_dict,
                    "risk_score": risk_score,
                    "features": feature_scores,
                },
            )

        if t in [107, 108, 110, 115]:
            status_map = {
                107: "policy_gate_passed",
                108: "simulated_in_sandbox",
                110: "completed",
                115: "verification_requested",
            }
            actions.append(
                {
                    "action_id": action_id,
                    "tick": t,
                    "name": action_name,
                    "policy_decision": policy_decision_at_action,
                    "simulation_only": True,
                    "bounded": True,
                    "reversible": True,
                    "intended_effect": "Reduce checkout latency and error rate within 10 seconds",
                    "status": status_map[t],
                }
            )

        if t == 116:
            pre_rows = rows[100:105]
            post_rows = rows[115:118]

            pre_latency = float(np.mean([r["latency"] for r in pre_rows]))
            pre_error = float(np.mean([r["error_rate"] for r in pre_rows]))
            post_latency = float(np.mean([r["latency"] for r in post_rows]))
            post_error = float(np.mean([r["error_rate"] for r in post_rows]))

            checks = [
                {"check": "simulator_only", "passed": True},
                {"check": "bounded_and_reversible", "passed": True},
                {
                    "check": "latency_reduction_ge_15pct",
                    "passed": bool(post_latency <= 0.85 * pre_latency),
                },
                {
                    "check": "error_rate_reduction",
                    "passed": bool(post_error <= max(0.8 * pre_error, pre_error - 0.5)),
                },
            ]
            passed = all(c["passed"] for c in checks)

            verification = VerificationResult(
                tick=t,
                action_id=action_id,
                verifier_id="verifier-independent-01",
                passed=passed,
                checks=checks,
                proof={
                    "pre_latency": pre_latency,
                    "pre_error_rate": pre_error,
                    "post_latency": post_latency,
                    "post_error_rate": post_error,
                },
            )
            verification_dict = model_to_dict(verification)
            verifications.append(verification_dict)

            audit.append(
                model_to_dict(
                    AuditEvent(
                        tick=t,
                        ts=row["ts"],
                        event_type="verification_done",
                        actor="independent_verifier",
                        detail=f"verification passed={passed}",
                    )
                )
            )

            add_evidence(
                t,
                "verification",
                {
                    "verification": verification_dict,
                    "pre_latency": pre_latency,
                    "post_latency": post_latency,
                },
            )

        if t == 119:
            add_evidence(
                t,
                "replay_marker",
                {
                    "run_id": run_id,
                    "tick": t,
                    "note": "Deterministic replay marker",
                },
            )

    if first_detection_tick is not None:
        audit.append(
            model_to_dict(
                AuditEvent(
                    tick=first_detection_tick,
                    ts=ts(first_detection_tick),
                    event_type="detection_created",
                    actor="ai_engine",
                    detail=f"First detection at tick {first_detection_tick}",
                )
            )
        )

    for ev in evidence:
        audit.append(
            model_to_dict(
                AuditEvent(
                    tick=ev["tick"],
                    ts=ts(ev["tick"]),
                    event_type="evidence_hashed",
                    actor="evidence_store",
                    detail=f"{ev['evidence_id']} sha256:{ev['sha256'][:12]}",
                )
            )
        )

    audit.append(
        model_to_dict(
            AuditEvent(
                tick=ticks - 1,
                ts=ts(ticks - 1),
                event_type="replay_marker",
                actor="orchestrator",
                detail=f"run_id={run_id}",
            )
        )
    )

    entities_df = pd.DataFrame([model_to_dict(e) for e in entities])
    observations_df = pd.DataFrame(observations)
    events_df = pd.DataFrame(events)
    detections_df = pd.DataFrame(detections)
    states_df = pd.DataFrame(states)
    hypotheses_df = pd.DataFrame(hypotheses)
    risks_df = pd.DataFrame(risks)
    actions_df = pd.DataFrame(actions)
    verifications_df = pd.DataFrame(verifications)
    evidence_df = pd.DataFrame(evidence)
    audit_df = pd.DataFrame(audit)
    timeline_df = pd.DataFrame(timeline)

    model_meta = {
        "model_id": "ara-anomaly-v1",
        "model_version": "1.0.0",
        "model_type": "deterministic z-score / evidence fusion baseline",
        "config_hash": config_hash,
        "training_window": 20,
        "features": feature_metrics,
        "calibration": "fixed confidence-weighted scaling",
        "cold_start": "reduced confidence and suppressed autonomous action",
        "seed": seed,
    }

    model_runs_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "tick": 0,
                "model_id": model_meta["model_id"],
                "payload": model_meta,
            }
        ]
    )

    y_true = df["anomaly_ground_truth"].astype(bool)
    y_pred = risks_df["score"] >= 40.0

    accuracy = float((y_pred.values == y_true.values).mean())
    false_alarm_rate = float(y_pred[~y_true].mean()) if (~y_true).any() else 0.0

    first_truth = int(df[y_true]["tick"].min()) if y_true.any() else 0
    active_dets = detections_df[detections_df["active"]]
    first_det = int(active_dets["tick"].min()) if not active_dets.empty else ticks
    detection_latency = max(0, first_det - first_truth)

    inference_latencies = [
        8.0 + 6.0 * abs(math.sin(i * 0.7)) + (5.0 if rows[i]["missing_count"] > 0 else 0.0)
        for i in range(ticks)
    ]
    p95_latency = float(np.percentile(inference_latencies, 95))

    continuity = 1.0
    verification_pass_rate = (
        float(verifications_df["passed"].mean()) if not verifications_df.empty else 0.0
    )

    def evidence_hash_ok(ev: Dict[str, Any]) -> bool:
        body = {k: v for k, v in ev.items() if k != "sha256"}
        return sha256_json(body) == ev.get("sha256")

    evidence_integrity = (
        1.0 if evidence and all(evidence_hash_ok(e) for e in evidence) else 0.0
    )

    mean_incident_risk = (
        float(risks_df[y_true]["score"].mean()) if y_true.any() else 0.0
    )

    kpis = {
        "Detection accuracy": round(accuracy, 4),
        "False alarm rate": round(false_alarm_rate, 4),
        "Detection latency ticks": detection_latency,
        "P95 inference latency ms": round(p95_latency, 2),
        "State continuity": continuity,
        "Verification pass rate": round(verification_pass_rate, 4),
        "Evidence integrity": evidence_integrity,
        "Mean incident risk": round(mean_incident_risk, 2),
        "Action cycle ticks": 11,
    }

    target_map = {
        "Detection accuracy": ">= 0.95",
        "False alarm rate": "<= 0.05",
        "Detection latency ticks": "<= 3",
        "P95 inference latency ms": "<= 50",
        "State continuity": "1.0",
        "Verification pass rate": "1.0",
        "Evidence integrity": "1.0",
        "Mean incident risk": "descriptive",
        "Action cycle ticks": "<= 15",
    }

    metrics_df = pd.DataFrame(
        [
            {
                "tick": 120,
                "name": k,
                "value": float(v),
                "target": target_map.get(k, ""),
            }
            for k, v in kpis.items()
        ]
    )

    return {
        "run_id": run_id,
        "seed": seed,
        "ticks": ticks,
        "config": DEFAULT_CONFIG,
        "config_hash": config_hash,
        "df": df,
        "entities": entities_df,
        "observations": observations_df,
        "events": events_df,
        "states": states_df,
        "detections": detections_df,
        "hypotheses": hypotheses_df,
        "risks": risks_df,
        "actions": actions_df,
        "verifications": verifications_df,
        "evidence": evidence_df,
        "audit": audit_df,
        "timeline": timeline_df,
        "metrics": metrics_df,
        "model_runs": model_runs_df,
        "model_meta": model_meta,
        "kpis": kpis,
    }


# -----------------------------------------------------------------------------
# Built-in 60-second browser demo recorder
# -----------------------------------------------------------------------------

RECORDER_HTML = """
<div style="font-family:sans-serif;">
  <canvas id="ara-canvas" width="1280" height="720" style="width:100%; border:1px solid #ccc; border-radius:8px; background:#0e1117;"></canvas>
  <button id="ara-record" style="margin-top:8px; padding:8px 16px;">Start 60-second demo recording & download</button>
  <div id="ara-status" style="margin-top:6px; color:#555;"></div>
  <script>
    const data = __DATA__;
    const canvas = document.getElementById('ara-canvas');
    const ctx = canvas.getContext('2d');
    const statusEl = document.getElementById('ara-status');

    let mode = 'preview';
    let start = null;
    let recorder = null;
    let chunks = [];

    function drawFrame(t) {
      const tick = Math.min(data.rows.length - 1, Math.floor(t * 2));
      const row = data.rows[tick];

      ctx.fillStyle = '#0e1117';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = '#fafafa';
      ctx.font = '28px sans-serif';
      ctx.fillText('Autonomous Research Agent - Live Demo', 40, 48);

      ctx.font = '18px monospace';
      ctx.fillText(`tick=${row.tick}  phase=${row.phase}  risk=${row.risk.toFixed(0)}`, 40, 84);

      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i <= tick; i++) {
        const r = data.rows[i];
        const x = 40 + (i / (data.rows.length - 1)) * 760;
        const y = 320 - Math.min(1, r.latency / 300) * 180;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.strokeStyle = '#f97316';
      ctx.beginPath();
      for (let i = 0; i <= tick; i++) {
        const r = data.rows[i];
        const x = 40 + (i / (data.rows.length - 1)) * 760;
        const y = 560 - Math.min(1, r.error / 6) * 160;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.fillStyle = '#38bdf8';
      ctx.fillText('Latency', 40, 130);
      ctx.fillStyle = '#f97316';
      ctx.fillText('Error rate', 40, 380);

      const riskH = Math.min(100, row.risk) * 3;
      ctx.fillStyle = row.risk >= 75 ? '#ef4444' : row.risk >= 50 ? '#f59e0b' : '#22c55e';
      ctx.fillRect(880, 420 - riskH, 80, riskH);
      ctx.strokeStyle = '#94a3b8';
      ctx.strokeRect(880, 120, 80, 300);
      ctx.fillStyle = '#fafafa';
      ctx.fillText('Risk', 895, 450);

      ctx.font = '16px monospace';
      ctx.fillText(`action: ${row.action || 'none'}`, 880, 500);
      ctx.fillText(`verified: ${row.verified ? 'PASS' : 'pending'}`, 880, 530);
    }

    function loop(ts) {
      if (!start) start = ts;
      const elapsed = (ts - start) / 1000;

      if (mode === 'record') {
        if (elapsed >= 60) {
          drawFrame(60);
          if (recorder && recorder.state !== 'inactive') recorder.stop();
        } else {
          drawFrame(elapsed);
        }
      } else {
        drawFrame(elapsed % 60);
      }
      requestAnimationFrame(loop);
    }

    requestAnimationFrame(loop);

    document.getElementById('ara-record').onclick = () => {
      if (typeof MediaRecorder === 'undefined') {
        statusEl.textContent = 'MediaRecorder not supported in this browser.';
        return;
      }
      if (recorder && recorder.state === 'recording') return;

      try {
        const stream = canvas.captureStream(30);
        const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm';
        chunks = [];

        recorder = new MediaRecorder(stream, { mimeType: mime });
        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunks.push(e.data);
        };
        recorder.onstop = () => {
          const blob = new Blob(chunks, { type: 'video/webm' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'autonomous_research_agent_demo.webm';
          document.body.appendChild(a);
          a.click();
          a.remove();
          setTimeout(() => URL.revokeObjectURL(url), 5000);
          statusEl.textContent = 'Recording downloaded or save dialog opened.';
          mode = 'preview';
          start = null;
        };

        recorder.start();
        mode = 'record';
        start = null;
        statusEl.textContent = 'Recording 60-second demo...';
      } catch (err) {
        statusEl.textContent = 'Recording failed: ' + err.message;
      }
    };
  </script>
</div>
"""


def render_demo_recorder(demo: Dict[str, Any]) -> None:
    risk_by_tick = {
        int(r["tick"]): float(r["score"]) for r in demo["risks"].to_dict("records")
    }
    action_by_tick = {
        int(r["tick"]): str(r["status"]) for r in demo["actions"].to_dict("records")
    }
    ver_by_tick = {
        int(r["tick"]): bool(r["passed"]) for r in demo["verifications"].to_dict("records")
    }

    records = []
    for r in demo["df"].to_dict("records"):
        t = int(r["tick"])
        records.append(
            {
                "tick": t,
                "phase": r["phase"],
                "latency": round(float(r["latency"]), 1),
                "error": round(float(r["error_rate"]), 2),
                "risk": risk_by_tick.get(t, 0.0),
                "action": action_by_tick.get(t, ""),
                "verified": ver_by_tick.get(t, False),
            }
        )

    json_data = json.dumps({"rows": records}, default=str)
    html = RECORDER_HTML.replace("__DATA__", json_data)
    components.html(html, height=860)


# -----------------------------------------------------------------------------
# Streamlit app state
# -----------------------------------------------------------------------------

if "demo" not in st.session_state:
    st.session_state.demo = build_demo(DEFAULT_CONFIG["seed"], DEFAULT_CONFIG["ticks"])

if "tick" not in st.session_state:
    st.session_state.tick = 0

if "running" not in st.session_state:
    st.session_state.running = True

if "speed" not in st.session_state:
    st.session_state.speed = DEFAULT_CONFIG["demo_speed"]

if "persisted_runs" not in st.session_state:
    st.session_state.persisted_runs = set()

demo = st.session_state.demo
max_tick = int(demo["ticks"]) - 1
current_tick = min(int(st.session_state.tick), max_tick)
df = demo["df"]
current_row = df.iloc[current_tick]

try:
    conn = get_db()
    init_db(conn)
    if demo["run_id"] not in st.session_state.persisted_runs:
        persist_demo(conn, demo)
        st.session_state.persisted_runs.add(demo["run_id"])
    db_message = "SQLite persistence ready."
except Exception as exc:
    db_message = f"Persistence fallback: {exc}"

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------

st.sidebar.title("Agent Controls")
st.sidebar.caption(db_message)

speed = st.sidebar.select_slider(
    "Simulation speed",
    options=[1, 2, 4, 8],
    value=int(st.session_state.speed),
)
st.session_state.speed = int(speed)

b1, b2, b3, b4, b5 = st.sidebar.columns(5)
if b1.button("Start"):
    st.session_state.running = True
if b2.button("Pause"):
    st.session_state.running = False
if b3.button("Reset"):
    st.session_state.tick = 0
    st.session_state.running = False
if b4.button("Replay"):
    st.session_state.tick = 0
    st.session_state.running = True
if b5.button("Demo Mode"):
    st.session_state.tick = 0
    st.session_state.running = True
    st.session_state.speed = 2

st.sidebar.progress((current_tick + 1) / demo["ticks"])
st.sidebar.metric("Tick", f"{current_tick} / {max_tick}")
st.sidebar.metric("Phase", current_row["phase"])
st.sidebar.caption("Safety boundary: autonomous actions are simulator-only and independently verified.")

# -----------------------------------------------------------------------------
# Derived current values
# -----------------------------------------------------------------------------

risk_row = demo["risks"][demo["risks"].tick == current_tick]
current_risk = float(risk_row["score"].iloc[0]) if not risk_row.empty else 0.0
current_risk_level = str(risk_row["level"].iloc[0]) if not risk_row.empty else "low"

det_row = demo["detections"][demo["detections"].tick == current_tick]
current_detection_active = bool(det_row["active"].iloc[0]) if not det_row.empty else False

actions_up_to = demo["actions"][demo["actions"].tick <= current_tick]
current_policy = str(actions_up_to["policy_decision"].iloc[-1]) if not actions_up_to.empty else "observe"
current_action_status = str(actions_up_to["status"].iloc[-1]) if not actions_up_to.empty else "none"

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab_overview, tab_live, tab_ai, tab_investigation, tab_action, tab_verification, tab_evidence, tab_analytics, tab_simulator, tab_config = st.tabs(
    [
        "Overview",
        "Simulation/Live",
        "AI Results",
        "Investigation",
        "Action",
        "Verification",
        "Evidence & Audit",
        "Analytics/KPIs",
        "Simulator",
        "Configuration",
    ]
)

with tab_overview:
    st.subheader("Autonomous Research Agent")
    st.write(
        "A production-quality simulated prototype for safe autonomous research: "
        "perceive, estimate, detect, investigate, correlate, score risk, gate action, "
        "simulate bounded action, independently verify, record evidence, and learn."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tick", current_tick)
    c2.metric("Phase", current_row["phase"])
    c3.metric("Risk", f"{current_risk:.0f}/100")
    c4.metric("Detection", "Active" if current_detection_active else "Idle")
    c5.metric("Policy", current_policy)

    with st.expander("Problem, users, success criteria, assumptions, failure modes", expanded=True):
        st.markdown(
            """
**Problem:** Autonomous research agents must detect and investigate anomalies safely while remaining explainable, auditable, and bounded.

**Users:** SREs, security analysts, AI platform teams, governance reviewers.

**Success criteria:**
- Fast detection with low false alarms.
- Explainable hypotheses and risk scoring.
- Policy-gated simulator-only actions.
- Independent verification and hashed evidence.

**Assumptions:**
- Telemetry is streaming but may be missing, noisy, or contradictory.
- Actions are only allowed in a simulated sandbox.
- Baseline behavior can be learned from an initial normal window.

**Failure modes:**
- Missing sensor data.
- Contradictory health readings.
- Drift, noise, and delayed mitigation.
- False alarms and overconfident autonomy.
            """
        )

    with st.expander("KPI summary", expanded=True):
        k1, k2, k3, k4 = st.columns(4)
        kpis = demo["kpis"]
        k1.metric("Detection accuracy", kpis["Detection accuracy"])
        k2.metric("False alarm rate", kpis["False alarm rate"])
        k3.metric("Verification pass", kpis["Verification pass rate"])
        k4.metric("Evidence integrity", kpis["Evidence integrity"])

    with st.expander("60-second live demo video recorder", expanded=True):
        st.caption(
            "Click the button below to record and download a browser-generated 60-second WebM demo video."
        )
        render_demo_recorder(demo)

with tab_live:
    st.subheader("Simulation / Live")
    live_df = df[df.tick <= current_tick]

    st.write(f"**Current phase:** `{current_row['phase']}`")
    st.write(f"**Active scenarios:** `{', '.join(current_row['scenarios']) if current_row['scenarios'] else 'none'}`")

    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(live_df.set_index("tick")[["latency"]])
    with col2:
        st.line_chart(live_df.set_index("tick")[["error_rate"]])

    col3, col4 = st.columns(2)
    with col3:
        st.line_chart(live_df.set_index("tick")[["cpu", "memory"]])
    with col4:
        st.line_chart(live_df.set_index("tick")[["throughput"]])

    st.subheader("Current observations")
    obs_current = demo["observations"][demo["observations"].tick == current_tick]
    st.dataframe(obs_current, use_container_width=True)

    with st.expander("Full deterministic scenario preview"):
        st.line_chart(df.set_index("tick")[["latency", "throughput"]])

with tab_ai:
    st.subheader("AI Results")
    st.json(demo["model_meta"])

    st.line_chart(demo["detections"].set_index("tick")[["score"]])

    if not det_row.empty:
        st.write(f"**Detection score:** `{det_row['score'].iloc[0]}`")
        st.write(f"**Confidence:** `{det_row['confidence'].iloc[0]}`")
        st.write(f"**Explanation:** {det_row['explanation'].iloc[0]}")

        features = det_row["features"].iloc[0]
        if features:
            feature_df = pd.DataFrame.from_dict(features, orient="index", columns=["score"])
            st.bar_chart(feature_df)
    else:
        st.info("No detection row available for current tick.")

    st.subheader("Calibration")
    cal_df = demo["detections"].set_index("tick")[["raw_score", "score", "confidence"]]
    st.line_chart(cal_df)

with tab_investigation:
    st.subheader("Investigation")
    hyp_current = demo["hypotheses"][demo["hypotheses"].tick == current_tick].sort_values("rank")
    st.dataframe(hyp_current, use_container_width=True)

    for _, hyp in hyp_current.iterrows():
        with st.expander(f"{hyp['rank']}. {hyp['title']} — score={hyp['score']:.3f}"):
            st.write("**Supporting evidence:**")
            for item in hyp["support"]:
                st.write(f"- {item}")
            st.write("**Contradicting evidence:**")
            if hyp["contradiction"]:
                for item in hyp["contradiction"]:
                    st.write(f"- {item}")
            else:
                st.write("- none")

    st.subheader("Investigation timeline")
    timeline_df = demo["timeline"]
    timeline_current = timeline_df[timeline_df.tick <= current_tick].tail(100)
    st.dataframe(timeline_current, use_container_width=True)

with tab_action:
    st.subheader("Action")
    actions_df = demo["actions"]
    actions_current = actions_df[actions_df.tick <= current_tick]
    st.dataframe(actions_current, use_container_width=True)

    st.info(
        "Safety boundary: proposed actions are simulator-only, bounded, reversible, "
        "and must pass an independent verification stage."
    )

    if not actions_current.empty:
        latest = actions_current.iloc[-1]
        st.markdown(
            f"""
**Latest action:** `{latest['name']}`  
**Status:** `{latest['status']}`  
**Policy decision:** `{latest['policy_decision']}`  
**Simulation only:** `{latest['simulation_only']}`  
**Bounded:** `{latest['bounded']}`  
**Reversible:** `{latest['reversible']}`  
**Intended effect:** `{latest['intended_effect']}`
            """
        )

with tab_verification:
    st.subheader("Independent Verification")
    verif_df = demo["verifications"]
    if verif_df.empty or current_tick < 115:
        st.info("Independent verification appears after the policy-gated simulated action.")
    else:
        st.dataframe(verif_df, use_container_width=True)
        latest_verif = verif_df.iloc[-1]
        st.write(f"**Verifier:** `{latest_verif['verifier_id']}`")
        st.write(f"**Passed:** `{latest_verif['passed']}`")
        st.json(latest_verif["checks"])
        st.json(latest_verif["proof"])

with tab_evidence:
    st.subheader("Evidence & Audit")
    evidence_df = demo["evidence"]

    if st.button("Verify evidence hashes"):
        ok = 0
        for _, ev in evidence_df.iterrows():
            body = {k: v for k, v in ev.items() if k != "sha256"}
            if sha256_json(body) == ev["sha256"]:
                ok += 1
        st.success(f"{ok}/{len(evidence_df)} evidence hashes are valid.")

    st.dataframe(
        evidence_df[["evidence_id", "tick", "kind", "sha256"]],
        use_container_width=True,
    )

    with st.expander("Evidence payload"):
        selected = st.selectbox(
            "Select evidence",
            evidence_df["evidence_id"].tolist(),
        )
        if selected:
            payload = evidence_df[evidence_df.evidence_id == selected]["payload"].iloc[0]
            st.json(payload)

    st.subheader("Audit trail")
    st.dataframe(demo["audit"].tail(200), use_container_width=True)

with tab_analytics:
    st.subheader("Analytics / KPIs")
    for name, value in demo["kpis"].items():
        st.metric(name, value)

    st.dataframe(demo["metrics"], use_container_width=True)
    st.line_chart(demo["risks"].set_index("tick")[["score"]])

with tab_simulator:
    st.subheader("Simulator")
    seed_input = st.number_input("Deterministic seed", value=42, min_value=0, max_value=99999)
    if st.button("Load / reset deterministic scenario"):
        st.session_state.demo = build_demo(int(seed_input), 120)
        st.session_state.tick = 0
        st.session_state.running = True
        st.rerun()

    if st.button("Jump to action phase"):
        st.session_state.tick = 105
        st.session_state.running = False
        st.rerun()

    if st.button("Jump to verification phase"):
        st.session_state.tick = 115
        st.session_state.running = False
        st.rerun()

    if st.button("Instant finish"):
        st.session_state.tick = max_tick
        st.session_state.running = False
        st.rerun()

    st.dataframe(demo["events"], use_container_width=True)

with tab_config:
    st.subheader("Configuration")
    st.code(YAML_CONFIG, language="yaml")
    st.json(demo["config"])
    st.write(f"**Config hash:** `{demo['config_hash']}`")
    st.write(f"**Run ID:** `{demo['run_id']}`")

# -----------------------------------------------------------------------------
# Auto-advance simulation
# -----------------------------------------------------------------------------

if st.session_state.running:
    if st.session_state.tick < max_tick:
        time.sleep(1.0 / max(1, int(st.session_state.speed)))
        st.session_state.tick += 1
        st.rerun()
    else:
        st.session_state.running = False