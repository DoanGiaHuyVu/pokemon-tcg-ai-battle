import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from cg.api import State, PlayerState, EnergyType, all_card_data, all_attack

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


# ─────────────────────────────────────────────────────────
# Core helper: energy matching
# ─────────────────────────────────────────────────────────

def _can_pay_cost(required_energies, attached_energies):
    """
    Returns True if `attached_energies` can satisfy `required_energies`.
    Colorless (0) in the requirement can be paid by any type.
    Any type of attached energy can pay a Colorless requirement.
    """
    required = list(required_energies)
    available = list(attached_energies)

    # First pass: match typed (non-Colorless) requirements
    typed_reqs = [r for r in required if r != EnergyType.COLORLESS]
    colorless_reqs = len(required) - len(typed_reqs)

    for req_type in typed_reqs:
        if req_type in available:
            available.remove(req_type)
        else:
            return False

    # Second pass: any remaining energy pays Colorless requirements
    return len(available) >= colorless_reqs


def _energy_shortfall(required_energies, attached_energies):
    """
    Returns how many more energy are needed to pay the cost.
    0 means the attack is ready. Positive means energy-short.
    """
    required = list(required_energies)
    available = list(attached_energies)

    typed_reqs = [r for r in required if r != EnergyType.COLORLESS]
    colorless_reqs = len(required) - len(typed_reqs)

    shortfall = 0
    for req_type in typed_reqs:
        if req_type in available:
            available.remove(req_type)
        else:
            shortfall += 1

    # Remaining available can cover colorless
    colorless_remaining = max(0, colorless_reqs - len(available))
    shortfall += colorless_remaining
    return shortfall


# ─────────────────────────────────────────────────────────
# Core helper: Pokémon attack analysis
# ─────────────────────────────────────────────────────────

def _get_active_hp_info(player_state):
    """Returns (current_hp, max_hp, damage_on_active) or None if no active."""
    if not player_state.active or player_state.active[0] is None:
        return None
    pkmn = player_state.active[0]
    damage = pkmn.maxHp - pkmn.hp
    return pkmn.hp, pkmn.maxHp, damage


def _get_pokemon_attacks(pokemon):
    """Returns list of (Attack, shortfall) for a Pokémon, sorted by damage desc."""
    _ensure_metadata()
    card_data = _card_table.get(pokemon.id)
    if card_data is None:
        return []

    attached = list(pokemon.energies) if pokemon.energies else []
    results = []
    for atk_id in card_data.attacks:
        atk = _attack_table.get(atk_id)
        if atk is None:
            continue
        sf = _energy_shortfall(atk.energies, attached)
        results.append((atk, sf))

    # Sort by shortfall ascending (closest to ready first), then damage descending
    results.sort(key=lambda x: (x[1], -x[0].damage))
    return results


def _best_available_attack_damage(player_state):
    """Returns the best attack damage the active can deal RIGHT NOW, or 0."""
    if not player_state.active or player_state.active[0] is None:
        return 0
    pkmn = player_state.active[0]
    attacks = _get_pokemon_attacks(pkmn)
    for atk, sf in attacks:
        if sf == 0:
            return atk.damage
    return 0


def _can_active_attack(player_state):
    """Returns True if the active Pokémon has enough energy to use at least one attack."""
    return _best_available_attack_damage(player_state) > 0


def _min_shortfall_active(player_state):
    """Returns the minimum energy shortfall for any attack on the active, or 999 if no active."""
    if not player_state.active or player_state.active[0] is None:
        return 999
    pkmn = player_state.active[0]
    attacks = _get_pokemon_attacks(pkmn)
    if not attacks:
        return 999
    return min(sf for _, sf in attacks)


def _min_shortfall_bench(player_state):
    """Returns the minimum energy shortfall across all benched Pokémon, or 999."""
    best = 999
    for pkmn in (player_state.bench or []):
        attacks = _get_pokemon_attacks(pkmn)
        for _, sf in attacks:
            if sf < best:
                best = sf
    return best


def _count_bench_pokemon(player_state):
    """Returns the count of benched Pokémon."""
    return len(player_state.bench or [])


# ─────────────────────────────────────────────────────────
# score_state: Absolute board evaluation
# ─────────────────────────────────────────────────────────

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

    # --- 2. KO Proximity (damage ratio + threshold bonuses) ---
    opp_active_info = _get_active_hp_info(opp_state)
    my_active_info = _get_active_hp_info(my_state)

    our_best_atk = _best_available_attack_damage(my_state)
    opp_best_atk = _best_available_attack_damage(opp_state)

    if opp_active_info:
        opp_hp, opp_max_hp, opp_damage = opp_active_info
        damage_ratio = opp_damage / max(opp_max_hp, 1)
        score += damage_ratio * 200.0

        if opp_hp <= 0:
            score += 500.0  # Opponent active KO'd

        if our_best_atk > 0 and opp_hp <= our_best_atk:
            score += 250.0  # Opponent in KO range

    # ── Board Strength (Active + Bench) ──
    # We want to encourage high HP, evolved Pokemon, and attached energy everywhere!
    board_hp_score = 0.0
    board_energy_score = 0.0
    
    all_my_pokemon = []
    if my_state.active: all_my_pokemon.extend(my_state.active)
    if my_state.bench: all_my_pokemon.extend(my_state.bench)
    
    for p in all_my_pokemon:
        if not p: continue
        # HP score
        max_hp = getattr(p, 'hp', 70)
        dmg = getattr(p, 'damage', 0)
        board_hp_score += (max_hp - dmg) * 0.5
        
        # Evolution score
        stage = getattr(p, 'stage', 0)
        if stage == 1: board_hp_score += 30.0
        if stage == 2: board_hp_score += 100.0
        
        # Energy score
        energies = getattr(p, 'energies', [])
        energy_count = len(energies) if isinstance(energies, list) else energies
        board_energy_score += energy_count * 50.0  # Encourage attaching energy anywhere!

    score += board_hp_score
    score += board_energy_score
    
    my_active_info = _get_active_hp_info(my_state)
    if my_active_info:
        my_hp, my_max_hp, my_damage = my_active_info
        my_damage_ratio = my_damage / max(my_max_hp, 1)
        score -= my_damage_ratio * 200.0
        
        if my_hp <= 0:
            score -= 500.0  # Our active KO'd
            
        if opp_best_atk > 0 and my_hp <= opp_best_atk:
            score -= 250.0  # We're in KO range
            
    # --- 2.5 Bench Damage and Evolution (Dragapult / Deck Specifics) ---
    _ensure_metadata()
    # Reward damage placed on opponent's bench
    opp_bench_dmg = 0
    for c in (opp_state.bench or []):
        opp_bench_dmg += getattr(c, 'damage', 0)
    score += opp_bench_dmg * 2.0  # Every 10 damage = +20 score
    
    # Reward getting our Stage 1 / Stage 2 out
    for c in [my_state.active[0]] + (my_state.bench or []):
        if not c: continue
        if c.id in _card_table:
            stg = getattr(_card_table[c.id], 'stage', None)
            stg_val = stg.value if hasattr(stg, 'value') else stg
            if stg_val == 1:
                score += 30.0  # Stage 1
            elif stg_val == 2:
                score += 100.0 # Stage 2

    # --- 3. Attack Readiness (turns_to_attack model) ---
    active_shortfall = _min_shortfall_active(my_state)

    if active_shortfall == 0:
        # Active can attack right now. Reward scales with attack strength!
        base_readiness = 150.0 + (our_best_atk * 1.0)
        
        if opp_active_info and our_best_atk > 0 and opp_active_info[0] <= our_best_atk:
            score += base_readiness + 300.0  # Can KO now
        else:
            score += base_readiness          # Can attack now
    elif active_shortfall == 1:
        # Scale potential damage by a factor so we prefer building big attackers
        score += 80.0 + (our_best_atk * 0.5)
    elif active_shortfall == 2:
        score += 30.0 + (our_best_atk * 0.25)
    else:
        score -= 100.0  # No attacker is close to ready

    # Bench attacker readiness
    bench_shortfall = _min_shortfall_bench(my_state)
    if bench_shortfall == 0:
        score += 80.0   # Bench attacker ready (backup)
    elif bench_shortfall == 1:
        score += 40.0   # Bench attacker one away

    # --- 4. Light Board Development ---
    score += _count_bench_pokemon(my_state) * 10.0
    
    # Reward Total Energy on Board
    total_energy = sum(len(c.energies or []) for c in [my_state.active[0]] + (my_state.bench or []) if c)
    score += total_energy * 100.0

    # --- 5. Hand Quality (very conservative, capped) ---
    my_hand = my_state.handCount or 0
    hand_bonus = min(my_hand, 5) * 3.0  # Max +15
    score += hand_bonus

    return score


# ─────────────────────────────────────────────────────────
# score_action_delta: Action quality via board change
# ─────────────────────────────────────────────────────────

def score_action_delta(current_state, next_state, your_index: int) -> float:
    """
    Scores the quality of an action by computing the change in board evaluation,
    plus explicit guards for missed opportunities and setup failures.
    """
    current_score = score_state(current_state, your_index)
    next_score = score_state(next_state, your_index)
    delta = next_score - current_score

    my_current = current_state.players[your_index]
    my_next = next_state.players[your_index]
    opp_current = current_state.players[1 - your_index]

    could_attack_before = _can_active_attack(my_current)

    # Detect if turn ended (opponent is now the active player)
    turn_ended = False
    if hasattr(next_state, 'yourIndex'):
        turn_ended = (next_state.yourIndex != your_index)

    # ── Combat Rewards (Damage / KO) ──
    opp_next = next_state.players[1 - your_index]
    opp_info_before = _get_active_hp_info(opp_current)
    opp_info_after = _get_active_hp_info(opp_next)
    
    opp_id_before = opp_current.active[0].id if opp_current.active and opp_current.active[0] else None
    opp_id_after = opp_next.active[0].id if opp_next.active and opp_next.active[0] else None
    
    dmg_before = opp_info_before[2] if opp_info_before else 0
    dmg_after = opp_info_after[2] if opp_info_after else 0
    
    # Did we deal damage without KOing?
    if opp_id_before == opp_id_after and dmg_after > dmg_before:
        delta += (dmg_after - dmg_before) * 5.0  # e.g., 50 damage = +250 score
        
    # Did we KO the active?
    if opp_id_before and (opp_id_before != opp_id_after or not opp_info_after):
        delta += 1500.0  # Massive reward for scoring a KO!

    # ── Attack Enablement Bonus ──
    if not could_attack_before and _can_active_attack(my_next):
        delta += 150.0  # Action enabled our attack

    # ── Energy Shortfall Improvement ──
    sf_before = _min_shortfall_active(my_current)
    sf_after = _min_shortfall_active(my_next)
    if sf_before > 0 and sf_after < sf_before:
        # Energy attachment brought us closer to attacking
        if sf_after == 0:
            delta += 120.0  # Now attack-ready
        else:
            delta += 60.0   # Getting closer

    # ── Bench Attacker Improvement ──
    bench_sf_before = _min_shortfall_bench(my_current)
    bench_sf_after = _min_shortfall_bench(my_next)
    if bench_sf_before > 1 and bench_sf_after <= 1:
        delta += 80.0  # Benched attacker is now close to ready

    # ── Missed Setup Guards ──
    if turn_ended:
        bench_before = _count_bench_pokemon(my_current)
        bench_after = _count_bench_pokemon(my_next)

        # If bench is empty/sparse and we didn't bench anything
        if bench_before == 0 and bench_after == 0:
            delta -= 100.0  # No backup attacker
            
    return delta
