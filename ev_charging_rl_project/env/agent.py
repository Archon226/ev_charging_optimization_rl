# env/agent.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class EVAgent:
    agent_id: str
    model: str
    user_type: str                 # "Subscriber" | "Payg" | "Contactless"
    objective: str                 # "cost" | "time" | "hybrid"
    efficiency_Wh_per_km: float    # e.g., 160 Wh/km
    battery_kwh: float             # usable capacity
    soc: float                     # 0..100
    alpha_cost: float = 0.5        # only for hybrid objective

    # runtime/state
    total_minutes: float = 0.0
    total_cost: float = 0.0
    distance_driven_km: float = 0.0
    charging_sessions: List[Dict] = field(default_factory=list)
    route_history: List[Dict] = field(default_factory=list)

    def current_kwh(self) -> float:
        return (self.soc / 100.0) * self.battery_kwh

    def can_reach(self, distance_km: float, reserve_soc: float = 0.0) -> bool:
        needed_kwh = (distance_km * self.efficiency_Wh_per_km) / 1000.0
        end_soc = (self.current_kwh() - needed_kwh) / max(self.battery_kwh, 1e-9) * 100.0
        return end_soc >= reserve_soc

    def energy_needed_to(self, target_soc: float) -> float:
        target_soc = max(0.0, min(100.0, target_soc))
        delta_soc = max(0.0, target_soc - self.soc)
        return self.battery_kwh * (delta_soc / 100.0)

    def apply_drive(self, distance_km: float, avg_speed_kph: float) -> Dict:
        distance_km = max(0.0, float(distance_km))
        hours = distance_km / max(1e-6, float(avg_speed_kph))
        minutes = hours * 60.0
        kwh_used = (distance_km * self.efficiency_Wh_per_km) / 1000.0
        soc_drop = (kwh_used / self.battery_kwh) * 100.0

        self.soc = max(0.0, self.soc - soc_drop)
        self.total_minutes += minutes
        self.distance_driven_km += distance_km
        self.route_history.append({"type": "drive", "km": distance_km, "minutes": minutes, "soc_after": self.soc})
        return {"minutes": minutes, "kwh_used": kwh_used, "soc_after": self.soc}

    def apply_charge(self, delivered_kwh: float, minutes: float, cost_breakdown: Dict) -> None:
        soc_gain = (max(0.0, float(delivered_kwh)) / self.battery_kwh) * 100.0
        self.soc = min(100.0, self.soc + soc_gain)
        self.total_minutes += float(minutes)
        self.total_cost += float(cost_breakdown.get("total_cost", 0.0))
        self.charging_sessions.append(cost_breakdown)
        self.route_history.append({"type": "charge", "kwh": delivered_kwh, "minutes": minutes, "soc_after": self.soc})

    def session_reward(self, session_minutes: float, session_cost: float) -> float:
        if self.objective == "cost":
            return -float(session_cost)
        if self.objective == "time":
            return -float(session_minutes)
        # hybrid
        a = float(self.alpha_cost)
        return -(a * float(session_cost) + (1 - a) * float(session_minutes))
    
    def reset(self):
        self.total_minutes = 0.0
        self.total_cost = 0.0
        self.distance_driven_km = 0.0
        self.charging_sessions.clear()
        self.route_history.clear()

    def is_fully_charged(self) -> bool:
        return self.soc >= 99.5

    @property
    def max_soc(self) -> float:
        return 100.0

    @property
    def location(self):
        # TripSimulator owns real lat/lon; this is only for stub envs
        return (0.0, 0.0)

    def update_state(self, *_args, **_kwargs):
        pass

