"""
Pipeline Performance Benchmarking

Benchmark complete pipeline performance and generate reports.
"""

import time
import statistics
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import json

from src.drawing_intelligence.orchestration import PipelineOrchestrator
from src.drawing_intelligence.database import DatabaseManager
from src.drawing_intelligence.orchestration import CheckpointManager, RoutingEngine
from src.drawing_intelligence.llm.budget_controller import BudgetController
from ..fixtures.test_data_generator import TestDataGenerator
from ..utils.test_helpers import create_test_config


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    operation: str
    mean_time: float
    median_time: float
    std_dev: float
    min_time: float
    max_time: float
    iterations: int
    throughput: float  # items per second


class PipelineBenchmark:
    """Benchmark pipeline performance."""

    def __init__(self, output_dir: str = "tests/performance/results"):
        """
        Initialize benchmark.

        Args:
            output_dir: Directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup test environment
        self.config = create_test_config()
        self.db = DatabaseManager(":memory:")  # In-memory for speed

        checkpoint_dir = Path("tests/test_data/benchmark_checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(str(checkpoint_dir))

        self.budget_controller = BudgetController(
            daily_budget_usd=100.0, per_drawing_limit_usd=10.0, db_manager=self.db
        )

        self.routing_engine = RoutingEngine(self.config, self.budget_controller)

        self.orchestrator = PipelineOrchestrator(
            config=self.config,
            db=self.db,
            checkpoint_manager=self.checkpoint_manager,
            routing_engine=self.routing_engine,
        )

        self.results: List[BenchmarkResult] = []

    def benchmark_single_drawing(self, iterations: int = 10) -> BenchmarkResult:
        """
        Benchmark single drawing processing.

        Args:
            iterations: Number of iterations

        Returns:
            BenchmarkResult
        """
        print(f"\nBenchmarking single drawing processing ({iterations} iterations)...")

        # Create test PDF
        pdf_path = "tests/test_data/benchmark_drawing.pdf"
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            TestDataGenerator.create_mock_pdf(pdf_path)
        except ImportError:
            print("⚠️  reportlab not available, using mock data")
            return BenchmarkResult(
                operation="single_drawing",
                mean_time=0.0,
                median_time=0.0,
                std_dev=0.0,
                min_time=0.0,
                max_time=0.0,
                iterations=0,
                throughput=0.0,
            )

        times = []

        for i in range(iterations):
            start = time.time()

            try:
                result = self.orchestrator.process_drawing(pdf_path)
                elapsed = time.time() - start
                times.append(elapsed)

                print(f"  Iteration {i+1}/{iterations}: {elapsed:.3f}s")

            except Exception as e:
                print(f"  ⚠️  Iteration {i+1} failed: {e}")

        if not times:
            print("❌ All iterations failed")
            return BenchmarkResult(
                operation="single_drawing",
                mean_time=0.0,
                median_time=0.0,
                std_dev=0.0,
                min_time=0.0,
                max_time=0.0,
                iterations=0,
                throughput=0.0,
            )

        result = BenchmarkResult(
            operation="single_drawing",
            mean_time=statistics.mean(times),
            median_time=statistics.median(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times),
            iterations=len(times),
            throughput=1.0 / statistics.mean(times),
        )

        self.results.append(result)

        print(f"\n✅ Single Drawing Benchmark:")
        print(f"   Mean: {result.mean_time:.3f}s")
        print(f"   Median: {result.median_time:.3f}s")
        print(f"   Std Dev: {result.std_dev:.3f}s")
        print(f"   Min: {result.min_time:.3f}s")
        print(f"   Max: {result.max_time:.3f}s")
        print(f"   Throughput: {result.throughput:.2f} drawings/sec")

        return result

    def benchmark_batch_processing(
        self, batch_sizes: List[int] = [5, 10, 20], workers: int = 4
    ) -> Dict[int, BenchmarkResult]:
        """
        Benchmark batch processing with different sizes.

        Args:
            batch_sizes: List of batch sizes to test
            workers: Number of parallel workers

        Returns:
            Dict mapping batch size to BenchmarkResult
        """
        print(f"\nBenchmarking batch processing...")

        results = {}

        for batch_size in batch_sizes:
            print(f"\n  Testing batch size: {batch_size}")

            # Create test PDFs
            pdf_dir = Path(f"tests/test_data/batch_benchmark_{batch_size}")
            pdf_dir.mkdir(parents=True, exist_ok=True)

            pdf_paths = []
            for i in range(batch_size):
                pdf_path = pdf_dir / f"drawing_{i+1}.pdf"
                try:
                    if not pdf_path.exists():
                        TestDataGenerator.create_mock_pdf(str(pdf_path))
                    pdf_paths.append(str(pdf_path))
                except ImportError:
                    print(f"    ⚠️  Cannot create PDFs")
                    continue

            if not pdf_paths:
                continue

            # Benchmark
            start = time.time()

            try:
                batch_result = self.orchestrator.process_batch(
                    pdf_paths=pdf_paths,
                    batch_id=f"BENCH-{batch_size}",
                    parallel_workers=workers,
                )

                elapsed = time.time() - start

                result = BenchmarkResult(
                    operation=f"batch_{batch_size}",
                    mean_time=elapsed / batch_size,
                    median_time=elapsed / batch_size,
                    std_dev=0.0,
                    min_time=elapsed,
                    max_time=elapsed,
                    iterations=batch_size,
                    throughput=batch_size / elapsed,
                )

                results[batch_size] = result
                self.results.append(result)

                print(f"    ✅ Batch {batch_size}:")
                print(f"       Total Time: {elapsed:.3f}s")
                print(f"       Per Drawing: {result.mean_time:.3f}s")
                print(f"       Throughput: {result.throughput:.2f} drawings/sec")
                print(f"       Success Rate: {batch_result.success_rate * 100:.1f}%")

            except Exception as e:
                print(f"    ❌ Batch {batch_size} failed: {e}")

        return results

    def benchmark_component_stages(self) -> Dict[str, float]:
        """
        Benchmark individual pipeline stages.

        Returns:
            Dict mapping stage name to average time
        """
        print("\nBenchmarking individual stages...")

        # Create test data
        pdf_path = "tests/test_data/stage_benchmark.pdf"
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            TestDataGenerator.create_mock_pdf(pdf_path)
        except ImportError:
            print("⚠️  Cannot benchmark stages without reportlab")
            return {}

        # Process and extract timings
        result = self.orchestrator.process_drawing(pdf_path)

        stage_times = {}
        if hasattr(result, "processing_times"):
            stage_times = result.processing_times

        print("\n✅ Stage Timings:")
        for stage, time_val in sorted(stage_times.items()):
            print(f"   {stage}: {time_val:.3f}s")

        return stage_times

    def generate_report(self):
        """Generate benchmark report."""
        print("\n" + "=" * 60)
        print("BENCHMARK REPORT")
        print("=" * 60)

        # Summary statistics
        if self.results:
            print("\nSummary:")
            for result in self.results:
                print(f"\n{result.operation}:")
                print(f"  Mean Time: {result.mean_time:.3f}s")
                print(f"  Throughput: {result.throughput:.2f} items/sec")
                print(f"  Iterations: {result.iterations}")

        # Save to JSON
        output_file = self.output_dir / f"benchmark_{int(time.time())}.json"

        report_data = {
            "timestamp": time.time(),
            "results": [
                {
                    "operation": r.operation,
                    "mean_time": r.mean_time,
                    "median_time": r.median_time,
                    "std_dev": r.std_dev,
                    "min_time": r.min_time,
                    "max_time": r.max_time,
                    "iterations": r.iterations,
                    "throughput": r.throughput,
                }
                for r in self.results
            ],
        }

        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\n✅ Report saved to: {output_file}")
        print("=" * 60)


def run_full_benchmark():
    """Run complete benchmark suite."""
    print("=" * 60)
    print("PIPELINE PERFORMANCE BENCHMARK")
    print("=" * 60)

    benchmark = PipelineBenchmark()

    # Benchmark single drawing
    benchmark.benchmark_single_drawing(iterations=5)

    # Benchmark batch processing
    benchmark.benchmark_batch_processing(batch_sizes=[3, 5], workers=2)

    # Benchmark stages
    benchmark.benchmark_component_stages()

    # Generate report
    benchmark.generate_report()


if __name__ == "__main__":
    run_full_benchmark()
