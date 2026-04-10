#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import wandb
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import (
    EvidenceUnit,
    load_embeddings,
    get_paper_units,
    list_papers,
    list_models,
)
from src.cost import PenaltyCostProfiler
from src.make_importance import make_importance
from src.make_selection import make_selection
from src.evaluation import HierarchicalCoverageEvaluator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    log.info(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    log.debug(f"Config keys: {list(config.keys())}")
    log.debug(f"Budget: {config['budget_B']}, Gamma: {config['gamma']}, Tau: {config['tau']}")
    log.debug(f"Strategies: {config['strategies']}, Objectives: {config['objectives']}, Solvers: {config['solvers']}")
    return config


def apply_costs(units: List[EvidenceUnit], config: Dict) -> List[EvidenceUnit]:
    log.debug("Applying penalty costs to units")
    profiler = PenaltyCostProfiler(config)
    for unit in units:
        unit.cost = profiler.calculate_cost(unit.tokens, unit.text)
    costs = [u.cost for u in units]
    log.debug(f"Cost stats: min={min(costs)}, max={max(costs)}, mean={sum(costs)/len(costs):.1f}")
    return units


def print_eval_table(paper_id: str, results: List[tuple]) -> None:
    """Print formatted evaluation table for a paper."""
    if not results:
        return

    # Extract decision node names from first result
    first_metrics = results[0][1]
    decision_keys = sorted([k for k in first_metrics if k.startswith("decision_") and k.endswith("_coverage")])
    decision_names = [k.replace("decision_", "").replace("_coverage", "") for k in decision_keys]

    # Build header (including new selection quality metrics)
    headers = ["Config", "GR"] + decision_names + ["Avg_Dec", "Redund", "Density", "Volume"]

    # Build rows
    rows = []
    for config_name, metrics in results:
        row = [config_name, f"{metrics['global_gr_coverage']:.4f}"]
        for dk in decision_keys:
            row.append(f"{metrics[dk]:.4f}")
        row.append(f"{metrics['avg_decision_coverage']:.4f}")
        # Selection quality metrics
        row.append(f"{metrics['redundancy_score']:.4f}")
        row.append(f"{metrics['avg_density_phi']:.4f}")
        row.append(f"{metrics['semantic_volume']:.2f}")
        rows.append(row)

    # Calculate column widths
    col_widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]

    # Print table
    log.info(f"\nEvaluation Results for: {paper_id}")
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    log.info(header_line)
    log.info("-" * len(header_line))
    for row in rows:
        log.info(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))))


def run_single_config(
    units: List[EvidenceUnit],
    config: Dict,
    strategy: str,
    objective: str,
    solver: str,
    paper_data: Dict,
    evaluator: HierarchicalCoverageEvaluator,
    paper_id: str,
) -> Dict[str, float]:
    import torch

    obj_display = "dpp" if solver == "dpp" else objective
    config_name = f"{strategy}/{obj_display}/{solver}"
    log.info(f"Running: strategy={strategy}, objective={obj_display}, solver={solver}")

    units_copy = deepcopy(units)

    # Compute importance scores
    log.debug("Computing importance scores")
    units_copy = make_importance(units_copy, config, strategy=strategy)
    imp_scores = [u.importance for u in units_copy]
    log.debug(f"Importance stats: min={min(imp_scores):.4f}, max={max(imp_scores):.4f}, mean={sum(imp_scores)/len(imp_scores):.4f}")

    # Run selection
    log.debug("Running selection")
    selected_units, selected_indices, total_cost, total_importance = make_selection(
        units_copy, config, objective=objective, solver=solver
    )

    # Selection metrics
    budget = config["budget_B"]
    budget_util = (total_cost / budget) * 100 if budget > 0 else 0
    coverage = (len(selected_units) / len(units_copy)) * 100
    avg_imp = total_importance / len(selected_units) if selected_units else 0

    log.info(f"  Selected: {len(selected_units)}/{len(units_copy)} units ({coverage:.1f}% coverage)")
    log.info(f"  Cost: {total_cost}/{budget} ({budget_util:.1f}% utilization)")
    log.info(f"  Importance: total={total_importance:.4f}, avg={avg_imp:.4f}")

    # Source breakdown
    source_counts = {}
    for u in selected_units:
        source_counts[u.source_type] = source_counts.get(u.source_type, 0) + 1
    log.debug(f"  Source breakdown: {source_counts}")
    log.debug(f"  Top 5 IDs: {[u.unit_id for u in selected_units[:5]]}")

    # Evaluation (use units_copy since indices are relative to filtered list)
    all_evidence = torch.stack([u.embedding for u in units_copy])
    selected_phi = torch.tensor([u.importance for u in selected_units], dtype=torch.float32)
    eval_metrics = evaluator.evaluate(paper_data, selected_indices, all_evidence, selected_phi)
    log.info(f"  Eval: gr_cov={eval_metrics['global_gr_coverage']:.4f}, avg_dec_cov={eval_metrics['avg_decision_coverage']:.4f}")
    log.info(f"  Selection Quality: redundancy={eval_metrics['redundancy_score']:.4f}, density={eval_metrics['avg_density_phi']:.4f}, volume={eval_metrics['semantic_volume']:.4f}")
    for key, val in eval_metrics.items():
        if key.startswith("decision_") and key.endswith("_coverage"):
            log.debug(f"    {key}: {val:.4f}")

    # Log to wandb
    wandb_metrics = {
        f"{config_name}/selected_units": len(selected_units),
        f"{config_name}/total_units": len(units_copy),
        f"{config_name}/coverage_pct": coverage,
        f"{config_name}/budget_utilization_pct": budget_util,
        f"{config_name}/total_cost": total_cost,
        f"{config_name}/total_importance": total_importance,
        f"{config_name}/avg_importance": avg_imp,
    }
    # Add all evaluation metrics
    for key, val in eval_metrics.items():
        wandb_metrics[f"{config_name}/{key}"] = val

    wandb.log(wandb_metrics, commit=False)

    return eval_metrics


def run_pipeline(
    config: Dict,
    model_name: str,
    paper_id: str,
    data: Dict,
    evaluator: HierarchicalCoverageEvaluator,
) -> List[tuple]:
    log.info("=" * 60)
    log.info(f"Paper: {paper_id}")
    log.info(f"Model: {model_name}")

    # Get paper data and units
    paper_data = data[model_name][paper_id]
    all_units = get_paper_units(data, model_name, paper_id, config)

    # Filter to only paper chunks for selection (requirements are evaluation targets, not evidence)
    units = [u for u in all_units if u.source_type == "paper"]

    # Embedding info
    emb_dim = units[0].embedding.shape[0] if units else 0
    log.info(f"Units: {len(units)} paper chunks (from {len(all_units)} total), Embedding dim: {emb_dim}")

    # Token stats
    tokens = [u.tokens for u in units]
    log.debug(f"Token stats: min={min(tokens)}, max={max(tokens)}, mean={sum(tokens)/len(tokens):.1f}, total={sum(tokens)}")

    # Apply costs
    units = apply_costs(units, config)
    total_available_cost = sum(u.cost for u in units)
    log.info(f"Total available cost: {total_available_cost}")

    # Get lists from config
    strategies = config["strategies"]
    objectives = config["objectives"]
    solvers = config["solvers"]

    # Loop over all combinations and collect results
    results = []
    for strategy in strategies:
        for solver in solvers:
            if solver == "dpp":
                metrics = run_single_config(units, config, strategy, "dpp", solver, paper_data, evaluator, paper_id)
                results.append((f"{strategy}/dpp/dpp", metrics))
            else:
                for objective in objectives:
                    metrics = run_single_config(units, config, strategy, objective, solver, paper_data, evaluator, paper_id)
                    results.append((f"{strategy}/{objective}/{solver}", metrics))

    # Print evaluation table for this paper
    print_eval_table(paper_id, results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Query-Independent Budgeted Retrieval Pipeline"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--model_name", type=str, required=True, help="Model name key in the .pt file")
    parser.add_argument("--paper_id", type=str, default=None, help="Paper ID to process (default: all)")
    parser.add_argument("--list_models", action="store_true", help="List available models and exit")
    parser.add_argument("--list_papers", action="store_true", help="List available papers and exit")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for metrics JSON files")
    parser.add_argument("--log_level", type=str, default="DEBUG", choices=["DEBUG", "INFO", "WARNING"], help="Logging level")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    log.info(f"Log level: {args.log_level}")

    # Load config
    config = load_config(args.config)

    # List mode (no wandb for list operations)
    if args.list_models or args.list_papers:
        data = load_embeddings(config)
        if args.list_models:
            models = list_models(data)
            log.info(f"Available models ({len(models)}):")
            for m in models:
                print(f"  - {m}")
        if args.list_papers:
            assert args.model_name, "--model_name required for --list_papers"
            papers = list_papers(data, args.model_name)
            log.info(f"Available papers for {args.model_name} ({len(papers)}):")
            for p in papers:
                print(f"  - {p}")
        return

    # Initialize wandb for pipeline run
    wandb.init(
        project="budgeted-retrieval",
        config={
            "model_name": args.model_name,
            "paper_id": args.paper_id,
            "budget_B": config["budget_B"],
            "gamma": config["gamma"],
            "tau": config["tau"],
            "num_clusters_k": config.get("num_clusters_k"),
            "distinctiveness_neighbors_m": config.get("distinctiveness_neighbors_m"),
            "prior_method": config.get("prior_method", "kmeans"),
            "lambda_mmr": config.get("lambda_mmr"),
            "lambda_fl": config.get("lambda_fl"),
            "dpp_quality_weight": config.get("dpp_quality_weight"),
            "strategies": config["strategies"],
            "objectives": config["objectives"],
            "solvers": config["solvers"],
        },
        settings=wandb.Settings(
            _disable_stats=True,  # Reduce background activity
            console="off",  # Don't capture console output (reduces sync load)
        ),
    )
    log.info(f"Wandb run initialized: {wandb.run.name}")

    # Run pipeline
    data = load_embeddings(config)
    paper_ids = [args.paper_id] if args.paper_id else list_papers(data, args.model_name)
    log.info(f"Processing {len(paper_ids)} paper(s)")

    # Create evaluator
    evaluator = HierarchicalCoverageEvaluator(config)

    # Collect all metrics across papers
    all_results: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    all_records: List[Dict] = []

    # Create wandb table for detailed results
    wandb_table = wandb.Table(columns=[
        "paper_id", "config", "strategy", "objective", "solver",
        "global_gr_coverage", "avg_decision_coverage",
        "redundancy_score", "avg_density_phi", "semantic_volume"
    ])

    for paper_idx, paper_id in enumerate(tqdm(paper_ids, disable=len(paper_ids) == 1)):
        paper_results = run_pipeline(config=config, model_name=args.model_name, paper_id=paper_id, data=data, evaluator=evaluator)
        for config_name, metrics in paper_results:
            all_results[config_name].append(metrics)
            # Store individual record
            all_records.append({
                "paper_id": paper_id,
                "config": config_name,
                "metrics": metrics,
            })
            # Add row to wandb table
            parts = config_name.split("/")
            wandb_table.add_data(
                paper_id,
                config_name,
                parts[0] if len(parts) > 0 else "",
                parts[1] if len(parts) > 1 else "",
                parts[2] if len(parts) > 2 else "",
                metrics.get("global_gr_coverage", 0),
                metrics.get("avg_decision_coverage", 0),
                metrics.get("redundancy_score", 0),
                metrics.get("avg_density_phi", 0),
                metrics.get("semantic_volume", 0),
            )
        # Commit wandb step after each paper
        wandb.log({"paper_idx": paper_idx, "paper_id": paper_id}, commit=True)

    # Compute aggregate metrics per config
    log.info("=" * 60)
    log.info("Computing aggregate metrics across all papers...")
    aggregate_metrics: Dict[str, Dict[str, float]] = {}

    for config_name, metrics_list in all_results.items():
        log.info(f"\nAggregate metrics for config: {config_name}")
        config_aggregates: Dict[str, float] = {}

        # Collect all unique metric keys across all papers
        if metrics_list:
            all_keys = set()
            for m in metrics_list:
                all_keys.update(m.keys())

            for key in sorted(all_keys):
                # Only average over papers that have this metric
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    avg_value = sum(values) / len(values)
                    config_aggregates[key] = avg_value
                    log.info(f"  {key}: {avg_value:.4f} (n={len(values)})")

        aggregate_metrics[config_name] = config_aggregates

        # Log aggregate metrics to wandb summary
        for key, val in config_aggregates.items():
            wandb.run.summary[f"aggregate/{config_name}/{key}"] = val

    # Log the detailed results table to wandb
    wandb.log({"detailed_results": wandb_table})

    # Prepare output data
    output_data = {
        "model_name": args.model_name,
        "num_papers": len(paper_ids),
        "paper_ids": paper_ids,
        "config_file": args.config,
        "aggregate_metrics": aggregate_metrics,
        "all_records": all_records,
    }

    # Save to JSON in output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    sanitized_model_name = args.model_name.replace("/", "_")
    output_file = output_dir / f"metrics_{sanitized_model_name}.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    log.info(f"\nMetrics saved to: {output_file}")

    # Log overall aggregate evaluation metrics from evaluator
    log.info("=" * 60)
    log.info("Overall aggregate metrics (across all configs and papers):")
    evaluator.counter.log_summary()

    # Log summary stats to wandb
    wandb.run.summary["num_papers"] = len(paper_ids)
    wandb.run.summary["num_configs"] = len(all_results)
    wandb.run.summary["total_records"] = len(all_records)

    # Finish wandb run and wait for sync to complete
    log.info("Finishing wandb run and syncing data...")
    wandb.finish(exit_code=0, quiet=False)
    log.info("Pipeline complete")


if __name__ == "__main__":
    main()
