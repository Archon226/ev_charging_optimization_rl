import numpy as np
from agent import EVAgent
from charger import Charger
from sumo.sumo_interface import SumoInterface
from reward import reward_cost_only, reward_time_only, reward_hybrid

class EVChargingEnv:
    def __init__(self, config):
        self.config = config
        self.num_agents = config.get("num_agents", 1)
        self.optimization_goal = config.get("goal", "hybrid")  # cost, time, or hybrid

        # SUMO Interface
        self.sumo = SumoInterface(config)

        # Load agents and chargers
        self.agents = [EVAgent(id=i) for i in range(self.num_agents)]
        self.chargers = self._load_chargers(config["charger_data"])

        # Internal state
        self.state = None

    def reset(self):
        """Start a new episode"""
        self.sumo.reset()

        for agent in self.agents:
            agent.reset()

        self.state = self._get_state()
        return self.state

    def step(self, action):
        """
        Accepts an action (e.g., index of charger) for agent[0]
        Returns: next_state, reward, done, info
        """
        agent = self.agents[0]
        selected_charger = self.chargers[action]

        self.sumo.route_agent_to_charger(agent, selected_charger)
        self.sumo.step()  # advance simulation

        agent.update_state(self.sumo)
        selected_charger.update_status(self.sumo)

        self.state = self._get_state()
        reward = self._compute_reward(agent)
        done = agent.is_fully_charged()

        return self.state, reward, done, {}

    def _get_state(self):
        """Returns a flat vector of the environment state"""
        agent = self.agents[0]
        loc_x, loc_y = agent.location
        soc = agent.soc / agent.max_soc
        time = self.sumo.get_time_of_day_norm()

        charger_availability = [1.0 if ch.is_available else 0.0 for ch in self.chargers]
        return np.array([loc_x, loc_y, soc] + charger_availability + [time], dtype=np.float32)

    def _compute_reward(self, agent):
        """Reward based on the goal type"""
        if self.optimization_goal == "cost":
            return reward_cost_only(agent)
        elif self.optimization_goal == "time":
            return reward_time_only(agent)
        else:
            return reward_hybrid(agent)

    def _load_chargers(self, charger_data):
        """Assumes charger_data is a list of charger dicts"""
        return [Charger(info) for info in charger_data]