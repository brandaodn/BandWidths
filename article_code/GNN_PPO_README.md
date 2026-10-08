# GNN + PPO prototype for matrix bandwidth minimization

This branch adds an experimental local-improvement policy while preserving the corrected DQN baseline.

## Design

1. Build a graph from the Matrix Market instance.
2. Generate the initial ordering with Reverse Cuthill-McKee (RCM).
3. Build per-node features:
   - normalized degree;
   - current normalized position;
   - RCM position;
   - normalized local bandwidth contribution.
4. Encode graph structure with a two-layer GCN implemented directly in PyTorch.
5. PPO scores every adjacent pair in the current ordering.
6. The selected action swaps one adjacent pair.
7. Reward is the normalized immediate bandwidth change.
8. The environment always records the best ordering visited.

This is intentionally a minimal prototype. It avoids PyTorch Geometric so the baseline can run with the repository's existing PyTorch dependency.

## Run

```bash
cd article_code
python gnn_ppo.py data/artigo_datas/bcsstk01.mtx --episodes 20 --steps 30
```

The output reports the RCM bandwidth, the best bandwidth found by PPO, the best ordering, and per-episode history.

## Scientific role

The corrected DQN remains the baseline. This branch tests a stronger formulation in which the policy acts directly on the ordering rather than merely selecting one centrality heuristic.
