"""
The learning agent.
Which variant it plays is chosen per match and can be configured using the Q_AGENT_CONFIG environment variable.

The variable must match an entry in the config.py file.

E.g. Q_AGENT_CONFIG=task2 python main.py play --no-gui --agents q_agent --train 1
"""
from agent_code.q_agent_gero import api

CONFIG = 'tournament'


def setup(self):
    api.setup(self, CONFIG)

def act(self, game_state: dict) -> str:
    return api.act(self, game_state)
