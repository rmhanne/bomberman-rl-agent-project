"""Training hooks"""

from typing import List

from agent_code.q_agent import api


def setup_training(self):
    api.setup_training(self)


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    api.game_events_occurred(self, old_game_state, self_action, new_game_state,
                             events)


def end_of_round(self, last_game_state: dict, last_action: str,
                 events: List[str]):
    api.end_of_round(self, last_game_state, last_action, events)
