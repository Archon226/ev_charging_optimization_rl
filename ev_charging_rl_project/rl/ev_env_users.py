from collections import defaultdict
import numpy as np
from simulator.candidates import find_charger_candidates
from simulator.trip_simulator import TripSimulator

class MultiEVChargingEnv:
    def __init__(self, sim, stations, connectors, pricing, ev_specs):
        self.sim = sim
        self.stations = stations
        self.connectors = connectors
        self.pricing = pricing
        self.ev_specs = ev_specs
        self.cache = defaultdict(list)
        self.simulators = []

    def reset(self, trips):
        self.simulators = []
        for trip in trips:
            sim = TripSimulator(self.sim, self.pricing)
            sim.reset(trip.origin, trip.dest, trip.ev_spec, trip.start_soc)
            self.simulators.append(sim)
        return self._get_states()

    def step(self, actions):
        rewards = []
        dones = []
        infos = []
        for sim, action in zip(self.simulators, actions):
            candidates = self.cache_or_find(sim)
            station = candidates[action]
            state, reward, done, info = sim.step(station)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return self._get_states(), rewards, dones, infos

    def cache_or_find(self, sim):
        key = (sim.origin, sim.dest, sim.ev_spec["model"])
        if key not in self.cache:
            self.cache[key] = find_charger_candidates(
                self.sim, sim.origin, sim.dest,
                self.stations, self.connectors, sim.ev_spec, sim.current_soc
            )
        return self.cache[key]

    def _get_states(self):
        return [{"soc": sim.current_soc, "pos": sim.position} for sim in self.simulators]
