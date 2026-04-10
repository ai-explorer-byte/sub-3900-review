"""
Importance scoring modules for query-independent budgeted retrieval.

Classes:
    - ImportancePriorScorer: K-Means clustering based importance scoring
    - CoresetImportanceScorer: Greedy Facility Location based importance scoring
"""

import logging
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from src.data_loader import EvidenceUnit

logger = logging.getLogger(__name__)


class ImportancePriorScorer:
    """
    K-Means clustering based importance scoring.
    
    Computes importance as: π_i = α_{ℓ_i} × softmax(e_i · μ_{ℓ_i} / τ)
    
    Where:
        - α_j = cluster distinctiveness (1 - mean similarity to top-m neighbors)
        - μ_j = cluster centroid (synthetic, geometric mean)
        - τ = softmax temperature
    """
    
    def __init__(self, config: Dict):
        self.num_clusters_k = config["num_clusters_k"]  # None means sqrt(n)
        self.m = config["distinctiveness_neighbors_m"]
        self.tau = config["tau"]

    def _compute_centroids(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = embeddings.shape[0]
        k = self.num_clusters_k if self.num_clusters_k is not None else int(math.sqrt(n))
        k = min(k, n)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(embeddings.numpy())
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=embeddings.dtype)
        labels = torch.tensor(kmeans.labels_, dtype=torch.long)
        return centroids, labels

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

    def _compute_unit_scores(
        self,
        embeddings: torch.Tensor,
        centroids: torch.Tensor,
        labels: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        centroids_norm = F.normalize(centroids, p=2, dim=1)

        # Similarity to all centroids: (n, k)
        sim_to_centroids = embeddings_norm @ centroids_norm.T

        # Softmax over centroids with temperature
        softmax_weights = F.softmax(sim_to_centroids / self.tau, dim=1)

        n = embeddings.shape[0]
        scores = torch.zeros(n)
        for i in range(n):
            assigned_k = labels[i].item()
            scores[i] = alpha[assigned_k] * softmax_weights[i, assigned_k]

        return scores

    def score(self, units: List[EvidenceUnit]) -> List[EvidenceUnit]:
        embeddings = torch.stack([u.embedding for u in units])

        # Check for NaN values and filter out affected units
        nan_mask = torch.isnan(embeddings).any(dim=1)
        if nan_mask.any():
            nan_indices = torch.where(nan_mask)[0].tolist()
            for idx in nan_indices:
                unit = units[idx]
                logger.warning(f"NaN embedding detected for unit index {idx}: {getattr(unit, 'id', 'unknown')}")
            logger.warning(f"Removing {len(nan_indices)} units with NaN embeddings from {len(units)} total")

            # Filter out units with NaN embeddings
            valid_mask = ~nan_mask
            units = [u for i, u in enumerate(units) if valid_mask[i]]
            embeddings = embeddings[valid_mask]

        centroids, labels = self._compute_centroids(embeddings)
        sim_matrix = self._compute_centroid_similarity(centroids)
        alpha = self._compute_distinctiveness(sim_matrix)
        scores = self._compute_unit_scores(embeddings, centroids, labels, alpha)

        for i, unit in enumerate(units):
            unit.importance = scores[i].item()

        return units


class CoresetImportanceScorer:
    """
    Greedy Facility Location (Coreset) based importance scoring.
    
    Replaces K-Means with Greedy Facility Location for representative selection.
    Representatives are actual data points rather than synthetic centroids.
    
    Computes importance as: π_i = α_{ℓ_i} × softmax(e_i · r_{ℓ_i} / τ)
    
    Where:
        - α_j = representative distinctiveness (1 - mean similarity to top-m neighbors)
        - r_j = representative embedding (actual data point)
        - τ = softmax temperature
    
    Key differences from ImportancePriorScorer:
        - Representatives are actual evidence units, not geometric means
        - Selection optimizes coverage (sum of distances) rather than variance
        - More robust to outliers and non-spherical cluster shapes
    """
    
    def __init__(self, config: Dict):
        self.num_clusters_k = config["num_clusters_k"]  # None means sqrt(n)
        self.m = config["distinctiveness_neighbors_m"]
        self.tau = config["tau"]

    def _select_coreset_representatives(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Greedy Facility Location selection (CRAIG-style).
        
        Selects k representative points that minimize total distance from all points
        to their nearest representative.
        
        Args:
            embeddings: (n, d) tensor of unit embeddings
            
        Returns:
            representatives: (k, d) tensor of selected embeddings (normalized)
            labels: (n,) tensor mapping each point to representative index 0..k-1
            S_indices: List of original indices of selected representatives
        """
        N = embeddings.shape[0]
        target_k = self.num_clusters_k if self.num_clusters_k is not None else int(math.sqrt(N))
        target_k = min(target_k, N)
        
        if target_k == 0:
            raise ValueError("target_k must be at least 1")
        
        # Normalize embeddings for distance computation
        # On L2-normalized vectors: d(x,y)² = 2(1 - cos(x,y))
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise Euclidean distances
        dists = torch.cdist(embeddings_norm, embeddings_norm, p=2)
        
        # Initialize: select most central point (minimizes sum of distances)
        first_idx = torch.argmin(dists.sum(dim=0)).item()
        
        S_indices = [first_idx]
        min_dists = dists[:, first_idx].clone()
        nearest_rep = torch.full((N,), first_idx, dtype=torch.long)
        
        # Greedy selection for remaining k-1 representatives
        for _ in range(1, target_k):
            # Marginal gain: reduction in total distance if we select point j
            # g(j|S) = Σ_i max(0, min_dists[i] - dists[i,j])
            reductions = torch.clamp(min_dists.unsqueeze(1) - dists, min=0)
            gains = reductions.sum(dim=0)
            
            # Mask already selected representatives
            for idx in S_indices:
                gains[idx] = -float('inf')
            
            best_idx = torch.argmax(gains).item()
            S_indices.append(best_idx)
            
            # Update minimum distances and assignments
            improved = dists[:, best_idx] < min_dists
            min_dists[improved] = dists[improved, best_idx]
            nearest_rep[improved] = best_idx
        
        # Convert actual indices to 0..k-1 labels
        idx_to_label = {idx: i for i, idx in enumerate(S_indices)}
        labels = torch.tensor(
            [idx_to_label[nearest_rep[i].item()] for i in range(N)], 
            dtype=torch.long
        )
        
        # Representatives are already normalized
        representatives = embeddings_norm[S_indices]
        
        return representatives, labels, S_indices

    def _compute_centroid_similarity(self, centroids: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine similarity between representatives."""
        centroids_norm = F.normalize(centroids, p=2, dim=1)
        sim_matrix = centroids_norm @ centroids_norm.T
        return sim_matrix

    def _compute_distinctiveness(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute distinctiveness α_j for each representative.
        
        α_j = 1 - mean(top-m similarities to other representatives)
        
        High α indicates the representative covers a semantically distinct region.
        """
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

    def _compute_unit_scores(
        self,
        embeddings: torch.Tensor,
        representatives: torch.Tensor,
        labels: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute importance score for each unit.
        
        π_i = α_{ℓ_i} × softmax(e_i · r_{ℓ_i} / τ)
        
        Where ℓ_i is the assigned representative for unit i.
        """
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        reps_norm = F.normalize(representatives, p=2, dim=1)

        # Similarity to all representatives: (n, k)
        sim_to_reps = embeddings_norm @ reps_norm.T

        # Softmax over representatives with temperature
        softmax_weights = F.softmax(sim_to_reps / self.tau, dim=1)

        n = embeddings.shape[0]
        scores = torch.zeros(n)
        for i in range(n):
            assigned_k = labels[i].item()
            scores[i] = alpha[assigned_k] * softmax_weights[i, assigned_k]

        return scores

    def score(self, units: List[EvidenceUnit]) -> List[EvidenceUnit]:
        """
        Score all units using coreset-based importance.
        
        Args:
            units: List of EvidenceUnit objects with embeddings
            
        Returns:
            Same list with importance scores assigned to each unit
        """
        if not units:
            return units
            
        embeddings = torch.stack([u.embedding for u in units])

        # Check for NaN values and filter out affected units
        nan_mask = torch.isnan(embeddings).any(dim=1)
        if nan_mask.any():
            nan_indices = torch.where(nan_mask)[0].tolist()
            for idx in nan_indices:
                unit = units[idx]
                logger.warning(
                    f"NaN embedding detected for unit index {idx}: "
                    f"{getattr(unit, 'id', 'unknown')}"
                )
            logger.warning(
                f"Removing {len(nan_indices)} units with NaN embeddings "
                f"from {len(units)} total"
            )

            # Filter out units with NaN embeddings
            valid_mask = ~nan_mask
            units = [u for i, u in enumerate(units) if valid_mask[i]]
            embeddings = embeddings[valid_mask]
        
        if len(units) == 0:
            return units

        # Core algorithm: Facility Location + Distinctiveness scoring
        representatives, labels, S_indices = self._select_coreset_representatives(embeddings)
        sim_matrix = self._compute_centroid_similarity(representatives)
        alpha = self._compute_distinctiveness(sim_matrix)
        scores = self._compute_unit_scores(embeddings, representatives, labels, alpha)

        for i, unit in enumerate(units):
            unit.importance = scores[i].item()

        logger.debug(
            f"CoresetImportanceScorer: scored {len(units)} units, "
            f"selected {len(S_indices)} representatives"
        )

        return units

## ==========================================
# 3. REAL DATA GENERATION (Using MPNet)
# ==========================================

def generate_large_real_document() -> List[EvidenceUnit]:
    print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # --- RAW TEXT DATA ---
    
    # 1. Backend (Technical - Distinct)
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
        "Continuous Deployment is managed via Jenkins pipelines that trigger on git tag creation."
    ]

    # 2. Frontend (Technical - Distinct)
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
        "Accessibility compliance is strictly monitored to meet WCAG 2.1 AA standards."
    ]

    # 3. Cloud (Technical - Distinct)
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
        "Content delivery is accelerated globally using Amazon CloudFront distributions."
    ]

    # 4. Database (Technical - Distinct)
    db_texts = [
        "The primary transactional database is a clustered PostgreSQL 15 setup.",
        "Connection pooling is managed by PgBouncer to handle high concurrency effectively.",
        "High-volume data tables are horizontally sharded across multiple physical nodes.",
        "Redis is used as a write-through cache to offload read traffic from the primary database.",
        "Analytical workloads are offloaded to a Snowflake data warehouse via nightly ETL jobs.",
        "Database backups are taken hourly and stored in immutable S3 buckets for ransomware protection.",
        "Schema changes are version-controlled and applied using Flyway migrations.",
        "We are evaluating a migration to Amazon Aurora Serverless for better scaling characteristics."
    ]

    # 5. NOISE (Boilerplate - Highly Repetitive Semantics)
    # These sentences mean almost the same thing, so MPNet will place them very close together.
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
        "This document is classified as Restricted / Internal."
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
            source_type="paper", # Generic for this example
            embedding=embeddings_tensor[i].cpu(), # Move to CPU for easier handling
            tokens=tokens,
            cost=tokens * 2,
            text=text,
            metadata={"topic": src_type}
        )
        units.append(unit)
        
    return units

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    print("--- 1. Generating Real Document Embeddings ---")
    try:
        units = generate_large_real_document()
    except ImportError:
        print("ERROR: Please install sentence-transformers: `pip install sentence-transformers`")
        return

    print(f"Generated {len(units)} units with real MPNet embeddings.")
    
    # CONFIGURATION
    # We expect roughly 5 clusters: Backend, Frontend, Cloud, DB, Legal.
    config = {
        "num_clusters_k": 10, 
        "distinctiveness_neighbors_m": 5, 
        "tau": 0.4 
    }
    
    print(f"\n--- 2. Running ImportancePriorScorer ---")
    print(f"Config: {config}")
    
    scorer = ImportancePriorScorer(config)
    scored_units = scorer.score(units)
    
    # Sort descending by score
    scored_units.sort(key=lambda x: x.importance, reverse=True)
    
    # OUTPUT RESULTS
    print("\n" + "="*100)
    print(f"{'RANK':<5} | {'SCORE':<8} | {'TOPIC':<10} | {'CONTENT'}")
    print("="*100)
    
    print("TOP 10 'MOST IMPORTANT' UNITS (The Distinct Signal):")
    for i in range(10):
        u = scored_units[i]
        topic = u.metadata.get('topic', 'unknown')
        print(f"#{i+1:<4} | {u.importance:.4f}   | {topic:<10} | {u.text[:60]}...")
        
    print("-" * 100)
    print("... (Middle units skipped) ...")
    print("-" * 100)

    print("BOTTOM 10 'LEAST IMPORTANT' UNITS (The Repetitive Noise):")
    for i in range(10):
        u = scored_units[-(10-i)] # Print from bottom up
        topic = u.metadata.get('topic', 'unknown')
        print(f"#{len(scored_units)-(9-i):<4} | {u.importance:.4f}   | {topic:<10} | {u.text[:60]}...")

if __name__ == "__main__":
    main()