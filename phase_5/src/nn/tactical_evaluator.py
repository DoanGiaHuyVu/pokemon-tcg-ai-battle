import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from cg.api import State, PlayerState, all_card_data, all_attack

# Pre-load card metadata for attack damage lookups
_all_cards = None
_card_table = None
_all_attacks = None
_attack_table = None

def _ensure_metadata():
    global _all_cards, _card_table, _all_attacks, _attack_table
    if _card_table is None:
        _all_cards = all_card_data()
        _card_table = {c.cardId: c for c in _all_cards}
    if _attack_table is None:
        _all_attacks = all_attack()
        _attack_table = {a.attackId: a for a in _all_attacks}


def _get_active_hp_info(player_state):
    """Returns (current_hp, max_hp, damage_on_active) or None if no active."""
    if not player_state.active or player_state.active[0] is None:
        return None
    pkmn = player_state.active[0]
    damage = pkmn.maxHp - pkmn.hp
    return pkmn.hp, pkmn.maxHp, damage


def _get_max_attack_damage(player_state):
    """Returns the maximum base attack damage the active Pokémon can deal, or 0."""
    _ensure_metadata()
    if not player_state.active or player_state.active[0] is None:
        return 0
    pkmn = player_state.active[0]
    card_data = _card_table.get(pkmn.id)
    if card_data is None:
        return 0
    
    max_dmg = 0
    for atk_id in card_data.attacks:
        atk = _attack_table.get(atk_id)
        if atk and atk.damage > max_dmg:
            max_dmg = atk.damage
    return max_dmg


def _can_active_attack(player_state):
    """Returns True if the active Pokémon has enough energy to use at least one attack."""
    _ensure_metadata()
    if not player_state.active or player_state.active[0] is None:
        return False
    pkmn = player_state.active[0]
    card_data = _card_table.get(pkmn.id)
    if card_data is None:
        return False
    
    attached_energies = list(pkmn.energies) if pkmn.energies else []
    
    for atk_id in card_data.attacks:
        atk = _attack_table.get(atk_id)
        if atk is None:
            continue
        # Check if we have enough energy for this attack
        required = list(atk.energies)
        available = list(attached_energies)
        can_use = True
        for req_type in required:
            if req_type in available:
                available.remove(req_type)
            elif 0 in available:
                # Colorless energy requirement — any type works
                available.remove(0)
            else:
                can_use = False
                break
        if can_use:
            return True
    return False


def _count_useful_energy(player_state):
    """Counts energy attached to active and bench Pokémon."""
    total = 0
    if player_state.active and player_state.active[0] is not None:
        total += len(player_state.active[0].energyCards or [])
    for bench_pkmn in (player_state.bench or []):
        total += len(bench_pkmn.energyCards or [])
    return total


def score_state(state, your_index: int) -> float:
    """
    Heuristically scores a game state from the perspective of `your_index`.
    Used both for absolute evaluation and for computing action deltas.
    """
    # Terminal outcomes dominate everything
    if state.result != -1:
        if state.result == your_index:
            return 10000.0
        else:
            return -10000.0

    my_state = state.players[your_index]
    opp_state = state.players[1 - your_index]
    score = 0.0

    # --- 1. Prize Lead (+300 per prize taken) ---
    my_prizes_taken = 6 - len(my_state.prize)
    opp_prizes_taken = 6 - len(opp_state.prize)
    prize_lead = my_prizes_taken - opp_prizes_taken
    score += prize_lead * 300.0

    # --- 2. KO Proximity (damage ratio, not raw damage) ---
    opp_active_info = _get_active_hp_info(opp_state)
    my_active_info = _get_active_hp_info(my_state)

    if opp_active_info:
        opp_hp, opp_max_hp, opp_damage = opp_active_info
        damage_ratio = opp_damage / max(opp_max_hp, 1)
        score += damage_ratio * 200.0  # Proportional damage reward

        if opp_hp <= 0:
            score += 500.0  # Opponent active KO'd

        # Check if opponent is within our next-attack KO range
        our_max_atk = _get_max_attack_damage(my_state)
        if our_max_atk > 0 and opp_hp <= our_max_atk:
            score += 250.0  # Opponent in KO range

    if my_active_info:
        my_hp, my_max_hp, my_damage = my_active_info
        my_damage_ratio = my_damage / max(my_max_hp, 1)
        score -= my_damage_ratio * 200.0  # Proportional damage penalty

        if my_hp <= 0:
            score -= 500.0  # Our active KO'd

        # Check if we're in opponent's KO range
        opp_max_atk = _get_max_attack_damage(opp_state)
        if opp_max_atk > 0 and my_hp <= opp_max_atk:
            score -= 250.0  # We're in KO range

    # --- 3. Attack Availability ---
    if _can_active_attack(my_state):
        score += 50.0  # We can attack

    # --- 4. Useful Energy Tempo ---
    my_energy = _count_useful_energy(my_state)
    opp_energy = _count_useful_energy(opp_state)
    score += my_energy * 20.0
    score -= opp_energy * 10.0

    # --- 5. Light Board Development ---
    score += len(my_state.bench or []) * 10.0

    # --- 6. Hand Quality (conservative) ---
    my_hand = my_state.handCount or 0
    score += min(my_hand, 7) * 3.0
    score -= max(0, 3 - my_hand) * 10.0  # Penalize near-empty hands

    return score


def score_action_delta(current_state, next_state, your_index: int) -> float:
    """
    Scores the quality of an action by computing the change in board evaluation.
    This is the primary scoring function used by the search agent.
    """
    current_score = score_state(current_state, your_index)
    next_score = score_state(next_state, your_index)
    delta = next_score - current_score

    # --- Missed Attack Guard ---
    # If we could attack before and still can after (meaning we didn't attack),
    # and the turn seems to have ended, penalize
    my_current = current_state.players[your_index]
    my_next = next_state.players[your_index]

    could_attack_before = _can_active_attack(my_current)
    can_still_attack = _can_active_attack(my_next)

    # Detect if turn advanced (opponent is now the active player)
    turn_ended = (next_state.yourIndex != your_index) if hasattr(next_state, 'yourIndex') else False

    if could_attack_before and turn_ended:
        # We had an attack available but ended our turn — missed attack
        delta -= 300.0

    # --- Missed KO Guard ---
    opp_current_info = _get_active_hp_info(current_state.players[1 - your_index])
    if opp_current_info:
        opp_hp_before = opp_current_info[0]
        our_max_atk = _get_max_attack_damage(my_current)
        if our_max_atk > 0 and opp_hp_before <= our_max_atk and turn_ended:
            # KO was available but we ended the turn
            opp_next_info = _get_active_hp_info(next_state.players[1 - your_index])
            if opp_next_info and opp_next_info[0] > 0:
                # Opponent is still alive — we missed the KO
                delta -= 700.0

    # --- Attack Enablement Bonus ---
    if not could_attack_before and _can_active_attack(my_next):
        delta += 120.0  # Action enabled our attack

    return delta
