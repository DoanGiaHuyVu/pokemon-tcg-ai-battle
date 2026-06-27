import os
import random

from cg.api import Observation, to_observation_class

def read_deck_csv() -> list[int]:
    file_path = os.path.join(os.path.dirname(__file__), "../decks/random_deck.csv")
    if not os.path.exists(file_path): file_path = "deck.csv"
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i].strip()))
    return deck

def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select == None:
        return read_deck_csv()
    
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)
