from centralities import centrality_heuristic


class Env:
    """Track the best ordering while keeping the original graph unchanged."""

    def __init__(self, opt: float):
        self.opt = float(opt)
        self.reset()

    def reset(self):
        self.n_steps = 0
        self.best_sol = self.opt
        self.best_graph = None
        self.best_solution = None
        self.previous_cost = self.opt
        self.gap = 0.0

    def get_initial_state(self):
        return [0.0, 0.0, 0.0, self.best_sol]

    def reward(self, current_cost: float, previous_best: float):
        """Return normalized improvement over the best solution so far."""
        if current_cost >= previous_best:
            return 0.0
        return (previous_best - current_cost) / max(1.0, abs(previous_best))

    def step(self, graph, centrality_values, cent_str, centralities):
        best_cost = float("inf")
        best_graph_step = None
        best_solution_step = None
        iterations = []

        for _ in range(5):
            cost, reordered_graph, solution, records = centrality_heuristic(
                graph=graph,
                centrality_values=centrality_values,
                cent_str=cent_str,
                alpha=0.3,
                iter_max=3,
                centralities=centralities,
            )
            iterations.extend(records)

            if cost < best_cost:
                best_cost = cost
                best_graph_step = reordered_graph
                best_solution_step = solution

        reward = self.reward(best_cost, self.best_sol)

        if best_cost < self.best_sol:
            self.best_sol = best_cost
            self.best_graph = best_graph_step
            self.best_solution = best_solution_step

        self.n_steps += 1
        self.previous_cost = best_cost
        self.gap = (self.opt - self.best_sol) / max(1.0, abs(self.opt))

        return {
            "n_steps": self.n_steps,
            "gap": self.gap,
            "reward": reward,
            "centrality": cent_str,
            "bandwidth": best_cost,
            "graph": self.best_graph,
            "solution": self.best_solution,
            "iterations": iterations,
        }
