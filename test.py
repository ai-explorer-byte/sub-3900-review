"""
Test script that simulates the complete selection process:
1. Generate test data with real embeddings (using SentenceTransformer)
2. Score units using ImportancePriorScorer (clustering + distinctiveness)
3. Select units using various objectives (MMR, GraphCut, FL) and solvers (greedy, cost_norm, topk, dpp)
4. Display results showing how different methods prioritize distinct vs. repetitive content
"""

import sys
from pathlib import Path

# Add src to path (like main.py)
sys.path.insert(0, str(Path(__file__).parent))

from sentence_transformers import SentenceTransformer

from src.data_loader import EvidenceUnit
from src.importance.importance_prior import ImportancePriorScorer
from src.make_selection import make_selection

import torch
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Any

class CoresetImportanceScorer:
    def __init__(self, config: Dict):
        self.num_clusters_k = config["num_clusters_k"]  # Target coreset size
        self.m = config["distinctiveness_neighbors_m"]
        self.tau = config["tau"]

    def _select_coreset_representatives(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replaces K-Means. Uses Greedy Facility Location (CRAIG) to pick k representative points.
        Returns:
            representatives: Tensor of shape (k, d)
            labels: Tensor of shape (n,) mapping each point to its nearest representative index (0 to k-1)
        """
        N = embeddings.shape[0]
        target_k = self.num_clusters_k if self.num_clusters_k is not None else int(math.sqrt(N))
        target_k = min(target_k, N)
        
        #normalize embeddings for distance computations
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 1. Compute Pairwise Distances (Euclidean)
        # Using simple broadcasting for clarity: dist_sq = |x|^2 + |y|^2 - 2<x,y>
        # (For production, torch.cdist is safer)
        dists = torch.cdist(embeddings, embeddings, p=2)

        # 2. Greedy Selection
        S_indices = []
        min_dists = torch.full((N,), float('inf'), dtype=dists.dtype)
        
        # To emulate cluster labels later, we track nearest selected index
        # We'll map the actual indices in S_indices to 0..k-1 later
        nearest_actual_idx = torch.zeros(N, dtype=torch.long)

        for _ in range(target_k):
            # Calculate Marginal Gain: How much does picking 'j' reduce total distance?
            # Broadcast min_dists (N, 1) vs dists (N, N)
            reductions = torch.clamp(min_dists.unsqueeze(1) - dists, min=0)
            gains = reductions.sum(dim=0)
            
            # Mask already selected
            if S_indices:
                gains[S_indices] = -1.0
            
            best_idx = torch.argmax(gains).item()
            S_indices.append(best_idx)
            
            # Update min distances
            improved_mask = dists[:, best_idx] < min_dists
            min_dists[improved_mask] = dists[improved_mask, best_idx]
            nearest_actual_idx[improved_mask] = best_idx

        # 3. Format as "Centroids" and "Labels"
        # The chosen points ARE the centroids now.
        representatives = embeddings[S_indices]
        
        # Map actual indices to 0..k-1 range for the scoring logic
        idx_map = {actual_idx: i for i, actual_idx in enumerate(S_indices)}
        labels = torch.tensor([idx_map[i.item()] for i in nearest_actual_idx], dtype=torch.long)

        return representatives, labels

    # --- BELOW IS THE EXACT ORIGINAL LOGIC ---

    def _compute_centroid_similarity(self, centroids: torch.Tensor) -> torch.Tensor:
        centroids_norm = F.normalize(centroids, p=2, dim=1)
        sim_matrix = centroids_norm @ centroids_norm.T
        return sim_matrix

    def _compute_distinctiveness(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        k = sim_matrix.shape[0]
        m = min(self.m, k - 1)
        alpha = torch.zeros(k)

        for i in range(k):
            sims = sim_matrix[i].clone()
            sims[i] = -float("inf")  # Exclude self-similarity
            if m > 0:
                topk_vals, _ = torch.topk(sims, m)
                alpha[i] = 1.0 - topk_vals.mean().item()
            else:
                alpha[i] = 1.0

        return alpha

    def _compute_unit_scores(self, embeddings, centroids, labels, alpha) -> torch.Tensor:
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        centroids_norm = F.normalize(centroids, p=2, dim=1)

        # Similarity to all representatives
        sim_to_centroids = embeddings_norm @ centroids_norm.T

        # Softmax over representatives
        softmax_weights = F.softmax(sim_to_centroids / self.tau, dim=1)

        n = embeddings.shape[0]
        scores = torch.zeros(n)
        for i in range(n):
            assigned_k = labels[i].item()
            # Original Formula: Distinctiveness * Probability
            scores[i] = alpha[assigned_k] * softmax_weights[i, assigned_k]

        return scores

    def score(self, units: List[EvidenceUnit]) -> List[EvidenceUnit]:
        if not units:
            return units
            
        embeddings = torch.stack([u.embedding for u in units])

        # Step 1: Use Coreset Selection instead of K-Means
        representatives, labels = self._select_coreset_representatives(embeddings)
        
        # Step 2: Use Original Logic for scoring
        sim_matrix = self._compute_centroid_similarity(representatives)
        alpha = self._compute_distinctiveness(sim_matrix)
        scores = self._compute_unit_scores(embeddings, representatives, labels, alpha)

        for i, unit in enumerate(units):
            unit.importance = scores[i].item()

        return units



# ==========================================
# 1. TEST DATA GENERATION
# ==========================================

def generate_test_data():
    """Generate test units with real embeddings using MPNet."""
    print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
    model = SentenceTransformer('all-mpnet-base-v2')

    # --- RAW TEXT DATA ---
    # 1. Backend (Technical - Distinct) -- expanded
    backend_texts = [
        "The backend utilizes a microservices architecture implemented in Python with FastAPI for high performance.",
        "We employ Consul for service discovery to ensure dynamic and reliable registration of new service instances.",
        "Inter-service communication is handled via gRPC to minimize latency compared to traditional REST APIs.",
        "Authentication is managed through an OAuth2 provider with JWT rotation for secure session handling.",
        "Asynchronous background tasks are processed using Celery workers backed by a RabbitMQ message broker.",
        "API rate limiting is enforced at the gateway level using a token bucket algorithm to prevent abuse.",
        "Legacy monolithic code is currently being strangled using the facade pattern into smaller services.",
        "Search functionality is powered by an Elasticsearch cluster configured with custom n-gram analyzers.",
        "Circuit breakers are implemented using Hystrix to prevent cascading failures across services.",
        "API documentation is automatically generated from code annotations following OpenAPI 3.0 standards.",
        "Unit test coverage is enforced at 87% in the CI pipeline before any merge is allowed.",
        "Continuous Deployment is managed via Jenkins pipelines that trigger on git tag creation.",
        "Service observability is achieved with Prometheus metrics and Grafana dashboards for real-time alerts.",
        "Request tracing is implemented with OpenTelemetry to correlate distributed request flows.",
        "Database migrations are orchestrated with a zero-downtime strategy using blue/green deployments.",
        "Feature flags are rolled out incrementally using a remote-config service to mitigate rollout risk.",
        "Connection retries use exponential backoff with jitter to avoid thundering herd effects.",
        "Traffic shaping and canary releases are managed at the ingress controller using weighted routing.",
        "Backend images are scanned for vulnerabilities as part of the container build pipeline.",
        "Secrets are injected at runtime via a sidecar that talks to a secure vault with short-lived tokens.",
        "Batch data processing jobs are scheduled on Kubernetes CronJobs with parallelism controls.",
        "Multipart uploads are coordinated through presigned URLs to offload upload traffic from app servers.",
        "A/B testing is supported by splitting users via consistent hashing of user IDs.",
        "Profiling is routinely performed in staging environments to catch CPU and memory hotspots."
    ]

    # 2. Frontend (Technical - Distinct) -- expanded
    frontend_texts = [
        "The user interface is a Single Page Application built with React 18 and TypeScript.",
        "Global state management is handled by Redux Toolkit to simplify data flow and reduce boilerplate.",
        "We utilize Next.js for Server-Side Rendering (SSR) to improve initial load times and SEO.",
        "CSS styling is standardized using Tailwind CSS to ensure design consistency across components.",
        "Component isolation testing is performed using Jest and React Testing Library.",
        "End-to-end regression tests are automated using Cypress running in headless CI environments.",
        "The application supports internationalization (i18n) for 12 distinct languages and locales.",
        "Webpack Module Federation is used to split the frontend into independently deployable micro-frontends.",
        "Real-time user performance metrics (Core Web Vitals) are tracked via Sentry integration.",
        "Accessibility compliance is strictly monitored to meet WCAG 2.1 AA standards.",
        "Client-side caching uses service workers to enable offline-first experiences for returning users.",
        "Image assets are served in AVIF/WEBP formats and resized on the fly to reduce bandwidth.",
        "Critical CSS is inlined for initial render while non-critical styles load asynchronously.",
        "Route-based code splitting reduces bundle sizes and speeds up perceived navigation.",
        "Form validation is implemented using a consistent schema-based library to avoid duplication.",
        "A design token system centralizes color, spacing, and typography across all components.",
        "Animation performance is optimized using transform and opacity changes over layout-triggering styles.",
        "Client telemetry captures slow-render incidents tied to specific component renders.",
        "Prefetching of likely next routes is used to make navigation feel instantaneous.",
        "Progressive enhancement ensures core functionality works without JavaScript for basic clients."
    ]

    # 3. Cloud (Technical - Distinct) -- expanded
    cloud_texts = [
        "Infrastructure is provisioned entirely as code (IaC) using Terraform modules.",
        "Containerized applications are orchestrated on Amazon EKS (Kubernetes) clusters.",
        "Auto-scaling groups are configured to scale out based on CPU utilization metrics.",
        "Secrets and API keys are managed via AWS Secrets Manager with automatic rotation policies.",
        "Centralized logging is implemented using the ELK stack (Elasticsearch, Logstash, Kibana).",
        "Event-driven processing for file uploads is handled by AWS Lambda functions triggered by S3 events.",
        "Network isolation is achieved using VPC Security Groups and strict Network ACLs.",
        "Cost optimization is managed by utilizing Spot Instances for stateless worker nodes.",
        "Disaster recovery strategy relies on Cross-Region Replication (CRR) of critical databases.",
        "Content delivery is accelerated globally using Amazon CloudFront distributions.",
        "Service mesh observability is enabled via Istio to capture service-level metrics and traces.",
        "Blob storage lifecycle policies archive cold data to reduce storage costs over time.",
        "Immutable infrastructure practices reduce configuration drift and simplify rollbacks.",
        "Policy-as-code enforces cloud governance via automated pre-deployment checks.",
        "Private endpoints are used for managed services to keep traffic within the VPC.",
        "Cluster autoscaler is tuned to balance pod scheduling and cloud cost efficiency.",
        "Image build pipelines produce minimal base images to reduce container attack surface.",
        "Cloud-native secrets scanning prevents accidental leakage in source-controlled configs.",
        "Multi-account tenancy limits blast radius and simplifies billing across teams.",
        "Managed database replicas are used to serve read-heavy workloads with low latency."
    ]

    # 4. Database (Technical - Distinct) -- expanded
    db_texts = [
        "The primary transactional database is a clustered PostgreSQL 15 setup.",
        "Connection pooling is managed by PgBouncer to handle high concurrency effectively.",
        "High-volume data tables are horizontally sharded across multiple physical nodes.",
        "Redis is used as a write-through cache to offload read traffic from the primary database.",
        "Analytical workloads are offloaded to a Snowflake data warehouse via nightly ETL jobs.",
        "Database backups are taken hourly and stored in immutable S3 buckets for ransomware protection.",
        "Schema changes are version-controlled and applied using Flyway migrations.",
        "We are evaluating a migration to Amazon Aurora Serverless for better scaling characteristics.",
        "Indexes are periodically rebuilt and analyzed to maintain query performance.",
        "Fine-grained row-level security policies restrict access to sensitive datasets.",
        "Materialized views are refreshed incrementally to speed up expensive aggregations.",
        "Query plans are monitored and anomalous regressions trigger automatic alerts.",
        "Temporal tables track historical changes for audit and rollback scenarios.",
        "Foreign key cascades are intentionally avoided in hot paths to reduce locking contention.",
        "Columnar stores are used for OLAP workloads to improve throughput on large scans.",
        "Write amplification is mitigated by batching commits in high-throughput ingestion paths."
    ]

    # 5. NOISE (Boilerplate - Highly Repetitive Semantics) -- expanded
    legal_texts = [
        "CONFIDENTIAL: This document contains proprietary information belonging to the company.",
        "Strictly Confidential: Do not distribute this report outside the authorized committee.",
        "All intellectual property rights contained herein are reserved by the authoring entity.",
        "This section is subject to the Non-Disclosure Agreement (NDA) signed by all parties.",
        "Unauthorized reproduction or distribution of this material is strictly prohibited.",
        "The views expressed in this document do not constitute a binding legal warranty.",
        "Please refer to the Master Services Agreement for full details on liability limitations.",
        "Data privacy compliance regarding this information is discussed in the GDPR addendum.",
        "For Internal Use Only. This document is the property of Project Titan.",
        "Property of Titan Corp. Do not forward or copy without express written permission.",
        "This page is intentionally left blank for administrative notes.",
        "Subject to the terms and conditions of the prevailing Non-Disclosure Agreement.",
        "Legal Disclaimer: Past performance is not indicative of future results.",
        "Copyright 2024 Titan Corp. All rights reserved globally.",
        "Any disputes arising from this document shall be resolved in Delaware courts.",
        "End of section. Confidentiality notice applies to all preceding content.",
        "Please recycle this report after use to maintain security hygiene.",
        "The information provided is for due diligence purposes only.",
        "No liability is assumed for errors or omissions in this technical report.",
        "This document is classified as Restricted / Internal.",
        "Distribution of this material is limited to approved recipients with access rights.",
        "This document may contain forward-looking statements that involve risks and uncertainties.",
        "All recipients must acknowledge the confidentiality obligations before accessing attachments.",
        "The content herein is subject to revision without prior notice to stakeholders.",
        "Acceptance of this document constitutes agreement to the stated terms and conditions.",
        "The recipient agrees to handle all data in accordance with company data protection policies.",
        "Unauthorized modification of this document voids any warranties offered by the issuer.",
        "References to third-party materials do not imply endorsement by the authoring party.",
        "This material is provided 'as is' and excludes any implied guarantees or representations.",
        "Retention of this document outside secure systems is discouraged without explicit permission.",
        "Controlled distribution list applies; contact legal counsel for permission to share further.",
        "The content may include trade secrets and should be redacted before public release.",
        "Failure to comply with confidentiality terms may result in contractual penalties.",
        "This notice applies to all digital and printed copies of the enclosed material.",
        "Sensitive sections are watermarked and tracked for audit purposes across the distribution chain.",
        "Any reproduction must include this confidentiality legend on each copy made.",
        "Third-party contractors must sign a separate NDA prior to receiving this document.",
        "Information in this document is intended solely for the addressee and their agents."
    ]

    all_texts = backend_texts + frontend_texts + cloud_texts + db_texts + legal_texts
    all_types = (["backend"] * len(backend_texts) +
                 ["frontend"] * len(frontend_texts) +
                 ["cloud"] * len(cloud_texts) +
                 ["db"] * len(db_texts) +
                 ["legal_noise"] * len(legal_texts))

    print(f"Encoding {len(all_texts)} chunks...")
    embeddings_tensor = model.encode(all_texts, convert_to_tensor=True)

    # Create Units
    units = []
    for i, (text, src_type) in enumerate(zip(all_texts, all_types)):
        u_id = f"unit_{i:03d}"
        tokens = len(text.split())

        unit = EvidenceUnit(
            unit_id=u_id,
            source_type="paper",
            embedding=embeddings_tensor[i].cpu(),
            tokens=tokens,
            cost=tokens * 2,  # Simple cost model
            text=text,
            metadata={"topic": src_type}
        )
        units.append(unit)

    return units


# ==========================================
# 2. DISPLAY RESULTS
# ==========================================

def display_results(title, selected_units, total_cost, total_importance, budget):
    """Display selection results in a formatted table, sorted by importance score."""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    print(f"Selected {len(selected_units)} units | Cost: {total_cost}/{budget} | Total Importance: {total_importance:.4f}")
    print("-"*100)

    # Count by topic
    topic_counts = {}
    for u in selected_units:
        topic = u.metadata.get('topic', 'unknown')
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    print(f"Topic Distribution: {topic_counts}")
    print("-"*100)
    print(f"{'#':<3} | {'Score':<8} | {'Topic':<12} | {'Content'}")
    print("-"*100)

    # Sort selected units by importance score (descending)
    sorted_units = sorted(selected_units, key=lambda u: u.importance, reverse=True)

    # Display ALL selected units
    for i, u in enumerate(sorted_units):
        topic = u.metadata.get('topic', 'unknown')
        print(f"{i+1:<3} | {u.importance:.4f}   | {topic:<12} | {u.text[:55]}...")


# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def run_scorer_pipeline(scorer_name: str, scorer, units: List[EvidenceUnit], selection_config: Dict):
    """Run the full scoring and selection pipeline for a given scorer."""
    from copy import deepcopy

    print("\n" + "#"*100)
    print(f"# SCORER: {scorer_name}")
    print("#"*100)

    # Deep copy units since scoring modifies them in place
    units_copy = deepcopy(units)

    # Score units
    print(f"\nScoring {len(units_copy)} units...")
    scored_units = scorer.score(units_copy)

    # Show score distribution by topic
    print("\nImportance Score Statistics by Topic:")
    for topic in ["backend", "frontend", "cloud", "db", "legal_noise"]:
        topic_units = [u for u in scored_units if u.metadata.get('topic') == topic]
        if topic_units:
            scores = [u.importance for u in topic_units]
            avg_score = sum(scores) / len(scores)
            print(f"  {topic:<12}: avg={avg_score:.4f}, min={min(scores):.4f}, max={max(scores):.4f}")

    # Run selection with different methods
    print("\n--- Running Selection Algorithms ---")

    test_cases = [
        ("MMR + Greedy", "mmr", "greedy"),
        ("GraphCut + Greedy", "graphcut", "greedy"),
        ("Facility Location + Greedy", "fl", "greedy"),
        ("MMR + Cost-Normalized", "mmr", "cost_norm"),
        ("Top-k (no diversity)", "mmr", "topk"),
        ("DPP Selection", "dpp", "dpp"),
    ]

    for title, objective, solver in test_cases:
        selected_units, _, cost, importance = make_selection(
            scored_units, selection_config, objective=objective, solver=solver
        )
        display_results(
            f"[{scorer_name}] {title}",
            selected_units, cost, importance, selection_config["budget_B"]
        )


def main():
    print("="*100)
    print("BUDGETED RETRIEVAL SELECTION TEST")
    print("Comparing: ImportancePriorScorer (K-Means) vs CoresetImportanceScorer (Greedy Facility Location)")
    print("="*100)

    # Step 1: Generate test data
    print("\n--- Step 1: Generating Test Data ---")
    units = generate_test_data()
    print(f"Generated {len(units)} units with real MPNet embeddings")

    # Display data distribution
    topic_counts = {}
    for u in units:
        topic = u.metadata.get('topic', 'unknown')
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    print(f"Data distribution: {topic_counts}")

    # Scorer config (shared)
    scorer_config = {
        "num_clusters_k": 10,
        "distinctiveness_neighbors_m": 5,
        "tau": 0.4
    }
    print(f"\nScorer config: {scorer_config}")

    # Selection config (shared)
    selection_config = {
        "budget_B": 150,
        "lambda_mmr": 0.7,
    }
    print(f"Selection config: budget={selection_config['budget_B']}, lambda_mmr={selection_config['lambda_mmr']}")

    # Step 2: Run with ImportancePriorScorer (K-Means based)
    kmeans_scorer = ImportancePriorScorer(scorer_config)
    run_scorer_pipeline("K-Means (ImportancePriorScorer)", kmeans_scorer, units, selection_config)

    # Step 3: Run with CoresetImportanceScorer (Greedy Facility Location based)
    coreset_scorer = CoresetImportanceScorer(scorer_config)
    run_scorer_pipeline("Coreset (CoresetImportanceScorer)", coreset_scorer, units, selection_config)

    # Step 4: Show comparison summary
    print("\n" + "="*100)
    print("SUMMARY: Comparing K-Means vs Coreset-based Importance Scoring")
    print("="*100)
    print("""
Key Differences:
- ImportancePriorScorer uses K-Means clustering to find centroids
- CoresetImportanceScorer uses Greedy Facility Location (CRAIG) to select representative points

Both scorers then:
1. Compute distinctiveness (alpha) based on inter-centroid/representative similarity
2. Assign importance = alpha[cluster] * softmax_weight[point, cluster]

Expected Behavior:
- Both should assign LOWER scores to 'legal_noise' (repetitive content)
- Both should assign HIGHER scores to distinct technical topics
- Coreset selection may produce different cluster assignments since it picks actual data points
  as representatives rather than computing mean centroids

Selection Methods:
- MMR/GraphCut: Penalize redundancy during selection
- Top-k: Pure importance-based (baseline)
- Cost-normalized: Prefers cheaper units with good importance
- DPP: Determinantal point process for quality-diversity trade-off
""")


if __name__ == "__main__":
    main()
