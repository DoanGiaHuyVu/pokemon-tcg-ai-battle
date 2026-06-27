class BeliefTracker:
    def __init__(self, opponent_deck_list):
        # Count all cards the opponent started with
        self.initial_deck_counts = {}
        for c in opponent_deck_list:
            self.initial_deck_counts[c] = self.initial_deck_counts.get(c, 0) + 1
            
        self.reset()
        
    def reset(self):
        self.seen_cards = {}
        self.opponent_hand_size = 0
        self.opponent_deck_size = 60
        self.opponent_prize_size = 6
        
    def update(self, obs):
        """
        Takes the observation dictionary and updates the counts of known cards 
        in the opponent's discard, bench, and active spots.
        """
        # Reset seen counts for this turn (recalculate from visible board)
        # A more advanced tracker would parse logs to catch cards played and returned to deck
        self.seen_cards = {}
        
        op_state = obs.get("current", {}).get("players", [{}, {}])[1]
        
        def count_visible(card):
            if card is None: return
            c_id = card.get("id")
            if c_id:
                self.seen_cards[c_id] = self.seen_cards.get(c_id, 0) + 1
                
        # Count discard
        for c in op_state.get("discard", []): count_visible(c)
        # Count active
        for c in op_state.get("active", []): count_visible(c)
        # Count bench
        for c in op_state.get("bench", []): count_visible(c)
            
        self.opponent_hand_size = op_state.get("handCount", 0)
        self.opponent_deck_size = op_state.get("deckCount", 0)
        self.opponent_prize_size = len(op_state.get("prize", []))
        
    def get_probability_in_hand(self, card_id):
        """
        Returns the rough probability that a specific card ID is currently in the opponent's hand.
        """
        total_initial = self.initial_deck_counts.get(card_id, 0)
        known_seen = self.seen_cards.get(card_id, 0)
        
        remaining_unseen = max(0, total_initial - known_seen)
        total_unknown_locations = self.opponent_hand_size + self.opponent_deck_size + self.opponent_prize_size
        
        if total_unknown_locations == 0:
            return 0.0
            
        # Probability = (Remaining copies / Total unknown slots) * Hand size
        # (Assuming uniform distribution across Hand, Deck, Prizes)
        prob_per_slot = remaining_unseen / total_unknown_locations
        prob_in_hand = prob_per_slot * self.opponent_hand_size
        
        return min(1.0, prob_in_hand)

# Example usage inside ObservationEncoder:
# p_boss = belief_tracker.get_probability_in_hand(BOSS_ORDERS_ID)
# global_features.append(p_boss)
