class MetricsRegistry:
    def __init__(self):
        self.reset()
        
    def reset(self):
        # search_begin tracking
        self.search_begin_attempts = 0
        self.search_begin_successes = 0
        self.search_begin_failures = 0
        self.search_begin_failure_reasons = {}
        
        # search_step tracking (per legal action evaluated)
        self.search_step_attempts = 0
        self.search_step_successes = 0
        self.search_step_failures = 0
        self.search_step_failure_reasons = {}
        
        # action evaluation tracking
        self.action_eval_successes = 0
        self.action_eval_failures = 0
        self.legal_actions_considered = 0
        
        # decision-level tracking
        self.total_decisions = 0
        self.search_decisions = 0       # decisions where search was used
        self.fallback_decisions = 0     # decisions where fallback was used
        self.fallback_reasons = {}
        self.trivial_decisions = 0      # only 1 legal action, no search needed
        self.multi_select_skips = 0     # maxCount > 1, skipped to heuristic
        
    # --- search_begin ---
    def log_begin_attempt(self):
        self.search_begin_attempts += 1
        
    def log_begin_success(self):
        self.search_begin_successes += 1
        
    def log_begin_failure(self, reason: str):
        self.search_begin_failures += 1
        if reason not in self.search_begin_failure_reasons:
            self.search_begin_failure_reasons[reason] = 0
        self.search_begin_failure_reasons[reason] += 1
        
    # --- search_step ---
    def log_step_attempt(self):
        self.search_step_attempts += 1
        
    def log_step_success(self):
        self.search_step_successes += 1
        
    def log_step_failure(self, reason: str):
        self.search_step_failures += 1
        if reason not in self.search_step_failure_reasons:
            self.search_step_failure_reasons[reason] = 0
        self.search_step_failure_reasons[reason] += 1
        
    # --- action evaluation ---
    def log_action_eval_success(self):
        self.action_eval_successes += 1
        
    def log_action_eval_failure(self):
        self.action_eval_failures += 1
        
    def log_legal_actions(self, count: int):
        self.legal_actions_considered += count
        
    # --- decision-level ---
    def log_search_decision(self):
        self.total_decisions += 1
        self.search_decisions += 1
        
    def log_fallback_decision(self, reason: str):
        self.total_decisions += 1
        self.fallback_decisions += 1
        if reason not in self.fallback_reasons:
            self.fallback_reasons[reason] = 0
        self.fallback_reasons[reason] += 1
        
    def log_trivial_decision(self):
        self.total_decisions += 1
        self.trivial_decisions += 1
        
    def log_multi_select_skip(self):
        self.total_decisions += 1
        self.multi_select_skips += 1
        
    def print_report(self):
        print("-" * 50)
        print("Search Integrity Report")
        print("-" * 50)
        
        # search_begin
        print(f"\n[search_begin]")
        print(f"  Attempts:     {self.search_begin_attempts}")
        print(f"  Successes:    {self.search_begin_successes}")
        print(f"  Failures:     {self.search_begin_failures}")
        if self.search_begin_attempts > 0:
            rate = (self.search_begin_successes / self.search_begin_attempts) * 100
            print(f"  Success Rate: {rate:.1f}%")
        if self.search_begin_failure_reasons:
            for r, c in self.search_begin_failure_reasons.items():
                print(f"    - {r}: {c}")
                
        # search_step
        print(f"\n[search_step]")
        print(f"  Attempts:     {self.search_step_attempts}")
        print(f"  Successes:    {self.search_step_successes}")
        print(f"  Failures:     {self.search_step_failures}")
        if self.search_step_attempts > 0:
            rate = (self.search_step_successes / self.search_step_attempts) * 100
            print(f"  Success Rate: {rate:.1f}%")
        if self.search_step_failure_reasons:
            for r, c in self.search_step_failure_reasons.items():
                print(f"    - {r}: {c}")
                
        # action evaluation
        print(f"\n[action_eval]")
        print(f"  Successes:    {self.action_eval_successes}")
        print(f"  Failures:     {self.action_eval_failures}")
        print(f"  Total Legal:  {self.legal_actions_considered}")
        if self.legal_actions_considered > 0:
            coverage = (self.action_eval_successes / self.legal_actions_considered) * 100
            print(f"  Coverage:     {coverage:.1f}%")
            
        # decisions
        print(f"\n[decisions]")
        print(f"  Total:           {self.total_decisions}")
        print(f"  Search-driven:   {self.search_decisions}")
        print(f"  Fallback:        {self.fallback_decisions}")
        print(f"  Trivial (1 opt): {self.trivial_decisions}")
        print(f"  Multi-select:    {self.multi_select_skips}")
        if self.total_decisions > 0:
            fb_rate = (self.fallback_decisions / self.total_decisions) * 100
            print(f"  Fallback Rate:   {fb_rate:.1f}%")
        if self.fallback_reasons:
            for r, c in self.fallback_reasons.items():
                print(f"    - {r}: {c}")
        print("-" * 50)

# Global singleton
agent_metrics = MetricsRegistry()
