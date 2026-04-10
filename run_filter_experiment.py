#!/usr/bin/env python3
"""
Corpus filtering experiment: measure impact of pre-filtering to top-k% by importance.

Given pre-computed importance scores, filter corpus to top-k%, re-run selection,
and compare coverage to full corpus baseline.

Runs across all embedding models and both prior methods (kmeans + coreset).
"""
import argparse
import csv
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import EvidenceUnit, load_embeddings, get_paper_units, list_papers, list_models
from src.cost import PenaltyCostProfiler
from src.make_importance import make_importance
from src.make_selection import make_selection
from src.evaluation.hierarchical_eval import _compute_coverage

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def apply_costs(units: List[EvidenceUnit], config: Dict) -> List[EvidenceUnit]:
    """Apply penalty costs to units."""
    profiler = PenaltyCostProfiler(config)
    for unit in units:
        unit.cost = profiler.calculate_cost(unit.tokens, unit.text)
    return units


def evaluate_against_full_corpus(
    paper_data: Dict,
    selected_units: List[EvidenceUnit],
    full_evidence: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate selected units with ideal computed from full (unfiltered) corpus.

    This ensures the denominator stays constant across all retention levels,
    making coverage numbers directly comparable.
    """
    device = full_evidence.device

    if selected_units:
        selected_evidence = torch.stack([u.embedding for u in selected_units]).to(device)
    else:
        selected_evidence = torch.empty(0, full_evidence.shape[1], device=device)

    # GR coverage against full corpus ideal
    gr_embeddings = paper_data["general_requirements"]["embeddings"].to(device)
    _, _, gr_coverage = _compute_coverage(gr_embeddings, full_evidence, selected_evidence)

    metrics: Dict[str, float] = {"global_gr_coverage": gr_coverage}
    decision_coverages: List[float] = []

    for decision_node, sr_section in paper_data["specific_requirements"].items():
        sr_embeddings = sr_section["embeddings"].to(device)
        _, _, node_coverage = _compute_coverage(sr_embeddings, full_evidence, selected_evidence)
        decision_coverages.append(node_coverage)
        metrics[f"decision_{decision_node}_coverage"] = node_coverage

    metrics["avg_decision_coverage"] = (
        sum(decision_coverages) / len(decision_coverages) if decision_coverages else 0.0
    )

    return metrics


def run_selection_on(
    units: List[EvidenceUnit],
    config: Dict,
    objective: str = "mmr",
    solver: str = "greedy",
) -> List[EvidenceUnit]:
    """Run selection and return the selected units."""
    selected_units, _, _, _ = make_selection(
        units, config, objective=objective, solver=solver
    )
    return selected_units


def filter_units_by_importance(
    units: List[EvidenceUnit], retain_pct: float
) -> List[EvidenceUnit]:
    """
    Filter units to top retain_pct% by importance score.

    Args:
        units: List of units with importance scores populated
        retain_pct: Percentage to retain (e.g., 0.10 for 10%)

    Returns:
        Filtered list of units (top-k by importance)
    """
    n_retain = max(1, int(len(units) * retain_pct))
    sorted_units = sorted(units, key=lambda u: u.importance, reverse=True)
    return sorted_units[:n_retain]


def run_filter_experiment(
    config: Dict,
    model_name: str,
    paper_id: str,
    data: Dict,
    retain_percentages: List[float],
    budget: int,
    objective: str = "mmr",
    solver: str = "greedy",
    strategy: str = "prior",
    prior_method: str = "kmeans",
) -> List[Dict]:
    """
    Run the corpus filtering experiment for a single paper.

    All coverage metrics are computed against the FULL corpus ideal so that
    the denominator is constant across retention levels.

    Returns list of result dicts, one per retain percentage (plus full corpus baseline).
    """
    log.info(f"Running filter experiment for paper: {paper_id}")

    # Get paper data and units
    paper_data = data[model_name][paper_id]
    all_units = get_paper_units(data, model_name, paper_id, config)

    # Filter to paper chunks only
    units = [u for u in all_units if u.source_type == "paper"]
    log.info(f"Total paper units: {len(units)}")

    # Apply costs
    units = apply_costs(units, config)

    # Set prior_method in config for importance scoring
    config_copy = deepcopy(config)
    config_copy["prior_method"] = prior_method
    config_copy["budget_B"] = budget

    # Compute importance scores
    units = make_importance(units, config_copy, strategy=strategy)

    # Full corpus evidence — fixed denominator for all evaluations
    full_evidence = torch.stack([u.embedding for u in units])

    # --- ORACLE: Run on full corpus ---
    log.info("Running oracle selection on full corpus...")
    oracle_selected = run_selection_on(units, config_copy, objective, solver)
    oracle_ids = {u.unit_id for u in oracle_selected}
    oracle_metrics = evaluate_against_full_corpus(paper_data, oracle_selected, full_evidence)
    log.info(f"Oracle selected {len(oracle_ids)} units")

    results = []

    # Add oracle (100%) result
    results.append({
        "retain_pct": 1.0,
        "retain_pct_display": "100%",
        "num_units": len(units),
        "gr_coverage": oracle_metrics["global_gr_coverage"],
        "avg_dec_coverage": oracle_metrics["avg_decision_coverage"],
        "oracle_retention": 1.0,  # 100% by definition
        "selected_count": len(oracle_ids),
    })

    # --- FILTERED: Run on each retain percentage ---
    for retain_pct in retain_percentages:
        log.info(f"Running with {retain_pct*100:.0f}% corpus retention...")

        # Filter units
        filtered_units = filter_units_by_importance(units, retain_pct)
        log.info(f"  Filtered to {len(filtered_units)} units")

        # Check how many oracle units are in filtered set
        filtered_ids = {u.unit_id for u in filtered_units}
        oracle_in_filtered = oracle_ids & filtered_ids
        oracle_retention = len(oracle_in_filtered) / len(oracle_ids) if oracle_ids else 0.0
        log.info(f"  Oracle units in filtered: {len(oracle_in_filtered)}/{len(oracle_ids)} ({oracle_retention*100:.1f}%)")

        # Re-run selection on filtered corpus
        filtered_selected = run_selection_on(filtered_units, config_copy, objective, solver)

        # Evaluate against full corpus ideal (same denominator as oracle)
        metrics = evaluate_against_full_corpus(paper_data, filtered_selected, full_evidence)

        results.append({
            "retain_pct": retain_pct,
            "retain_pct_display": f"{retain_pct*100:.0f}%",
            "num_units": len(filtered_units),
            "gr_coverage": metrics["global_gr_coverage"],
            "avg_dec_coverage": metrics["avg_decision_coverage"],
            "oracle_retention": oracle_retention,
            "selected_count": len(filtered_selected),
        })

    return results


def aggregate_results(all_paper_results: List[List[Dict]]) -> List[Dict]:
    """
    Aggregate results across multiple papers by averaging metrics.
    """
    if not all_paper_results:
        return []

    # Group by retain_pct
    by_retain = {}
    for paper_results in all_paper_results:
        for r in paper_results:
            pct = r["retain_pct"]
            if pct not in by_retain:
                by_retain[pct] = []
            by_retain[pct].append(r)

    # Average
    aggregated = []
    for pct in sorted(by_retain.keys(), reverse=True):
        records = by_retain[pct]
        n = len(records)
        aggregated.append({
            "retain_pct": pct,
            "retain_pct_display": records[0]["retain_pct_display"],
            "num_units": sum(r["num_units"] for r in records) / n,
            "gr_coverage": sum(r["gr_coverage"] for r in records) / n,
            "avg_dec_coverage": sum(r["avg_dec_coverage"] for r in records) / n,
            "oracle_retention": sum(r["oracle_retention"] for r in records) / n,
            "selected_count": sum(r["selected_count"] for r in records) / n,
            "num_papers": n,
        })

    return aggregated


def print_results_table(results: List[Dict], title: str = "Filter Experiment Results"):
    """Print formatted results table."""
    headers = ["Retained %", "Units", "Dec. Cov", "GR Cov"]
    rows = []
    for r in results:
        rows.append([
            r["retain_pct_display"],
            f"{r['num_units']:.0f}",
            f"{r['avg_dec_coverage']:.4f}",
            f"{r['gr_coverage']:.4f}",
        ])

    # Simple markdown table formatter
    col_widths = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def fmt_row(items):
        return "| " + " | ".join(item.ljust(col_widths[i]) for i, item in enumerate(items)) + " |"

    print(f"\n{title}")
    print("=" * 60)
    print(fmt_row(headers))
    print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
    for row in rows:
        print(fmt_row(row))
    print()


def save_results_csv(all_records: List[Dict], output_path: str):
    """Save all experiment records to a CSV file."""
    if not all_records:
        return
    fieldnames = list(all_records[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"Results saved to {output_path}")


# All embedding models to evaluate
ALL_MODELS = [
    "google/embeddinggemma-300m",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-m3",
    "NovaSearch/stella_en_1.5B_v5",
    "Qwen/Qwen3-Embedding-0.6B",
]

# Prior methods: centroid (kmeans) and coreset (facility location)
ALL_PRIOR_METHODS = ["kmeans", "coreset"]


def main():
    import yaml

    parser = argparse.ArgumentParser(description="Corpus filtering experiment across all models and prior methods")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Single model to run (default: all models)")
    parser.add_argument("--prior_method", type=str, default=None,
                        choices=["kmeans", "coreset"],
                        help="Single prior method (default: both kmeans and coreset)")
    parser.add_argument("--paper_id", type=str, default=None, help="Paper ID (default: all)")
    parser.add_argument("--budget", type=int, default=1000, help="Budget constraint")
    parser.add_argument("--retain_pcts", type=str, default="3,5,10,20,50",
                        help="Comma-separated retain percentages (e.g., '3,5,10')")
    parser.add_argument("--objective", type=str, default="mmr", help="Objective function")
    parser.add_argument("--solver", type=str, default="greedy", help="Solver type")
    parser.add_argument("--strategy", type=str, default="prior", help="Importance strategy")
    parser.add_argument("--output", type=str, default="filter_experiment_results.csv",
                        help="Output CSV path")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Parse retain percentages
    retain_pcts = [float(x) / 100 for x in args.retain_pcts.split(",")]

    # Load data once (shared across all models)
    print("Loading embeddings...")
    data = load_embeddings(config)

    # Determine models to run
    available_models = list_models(data)
    if args.model_name:
        models = [args.model_name]
    else:
        models = [m for m in ALL_MODELS if m in available_models]
        missing = [m for m in ALL_MODELS if m not in available_models]
        if missing:
            print(f"Warning: models not found in data: {missing}")

    # Determine prior methods to run
    prior_methods = [args.prior_method] if args.prior_method else ALL_PRIOR_METHODS

    print(f"Models: {models}")
    print(f"Prior methods: {prior_methods}")
    print(f"Retain percentages: {[f'{p*100:.0f}%' for p in retain_pcts]}")
    print(f"Budget: {args.budget} | Objective: {args.objective} | Solver: {args.solver}")
    print()

    # Collect all flat records for CSV
    all_csv_records: List[Dict] = []

    total_combos = len(models) * len(prior_methods)
    combo_bar = tqdm(total=total_combos, desc="Overall", position=0)

    for model_name in models:
        paper_ids = [args.paper_id] if args.paper_id else list_papers(data, model_name)

        for prior_method in prior_methods:
            combo_label = f"{model_name} / {prior_method}"
            combo_bar.set_postfix_str(combo_label)

            all_paper_results: List[List[Dict]] = []

            paper_bar = tqdm(
                paper_ids,
                desc=f"  {combo_label}",
                position=1,
                leave=False,
            )
            for paper_id in paper_bar:
                paper_bar.set_postfix_str(paper_id)
                results = run_filter_experiment(
                    config=config,
                    model_name=model_name,
                    paper_id=paper_id,
                    data=data,
                    retain_percentages=retain_pcts,
                    budget=args.budget,
                    objective=args.objective,
                    solver=args.solver,
                    strategy=args.strategy,
                    prior_method=prior_method,
                )
                all_paper_results.append(results)

                # Flatten per-paper results into CSV records
                for r in results:
                    all_csv_records.append({
                        "model": model_name,
                        "prior_method": prior_method,
                        "paper_id": paper_id,
                        **r,
                    })

            paper_bar.close()

            # Print aggregated summary for this model + prior_method combo
            if len(paper_ids) > 1:
                aggregated = aggregate_results(all_paper_results)
                print_results_table(
                    aggregated,
                    title=f"{model_name} | {prior_method} — Aggregated ({len(paper_ids)} papers)",
                )
            elif len(paper_ids) == 1:
                print_results_table(
                    all_paper_results[0],
                    title=f"{model_name} | {prior_method} — {paper_ids[0]}",
                )

            combo_bar.update(1)

    combo_bar.close()

    # Save all records to CSV
    save_results_csv(all_csv_records, args.output)
    print(f"\nDone. {len(all_csv_records)} records across {len(models)} models × {len(prior_methods)} prior methods.")


if __name__ == "__main__":
    main()
