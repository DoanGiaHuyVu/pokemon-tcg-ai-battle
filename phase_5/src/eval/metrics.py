class MetricsRegistry:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.search_attempt_count = 0
        self.search_success_count = 0
        self.search_exception_count = 0
        self.fallback_count = 0
        self.fallback_reasons = {}
        self.neural_policy_count = 0
        self.heuristic_count = 0
        
    def log_search_attempt(self):
        self.search_attempt_count += 1
        
    def log_search_success(self):
        self.search_success_count += 1
        
    def log_search_exception(self, error: str):
        print("LOGGING SEARCH EXCEPTION:", error)
        self.search_exception_count += 1
        self.log_fallback(error)
        
    def log_fallback(self, reason: str):
        self.fallback_count += 1
        if reason not in self.fallback_reasons:
            self.fallback_reasons[reason] = 0
        self.fallback_reasons[reason] += 1
        
    def log_decision_source(self, is_neural: bool):
        if is_neural:
            self.neural_policy_count += 1
        else:
            self.heuristic_count += 1
            
    def print_report(self):
        print("-" * 50)
        print("Search Integrity Report")
        print(f"Attempts:      {self.search_attempt_count}")
        print(f"Successes:     {self.search_success_count}")
        print(f"Exceptions:    {self.search_exception_count}")
        if self.search_attempt_count > 0:
            print(f"Success Rate:  {(self.search_success_count/self.search_attempt_count)*100:.1f}%")
        print(f"Fallbacks:     {self.fallback_count}")
        for r, count in self.fallback_reasons.items():
            print(f"  - {r}: {count}")
        print(f"Neural Source: {self.neural_policy_count}")
        print(f"Heur. Source:  {self.heuristic_count}")
        print("-" * 50)

# Global singleton
agent_metrics = MetricsRegistry()
