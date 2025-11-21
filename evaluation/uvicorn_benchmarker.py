# evaluation/uvicorn_benchmarker.py
# Fully automatic Uvicorn multi-worker benchmark (2025 edition)
# Auto-detects any PDF in evaluation/test_pdfs/ — works on Windows, Linux, macOS

import json
import time
import subprocess
import requests
import psutil
import statistics
import threading
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict
from datetime import datetime
import sys

@dataclass
class WorkerConfig:
    workers: int
    timeout: int = 300

    def to_cmd(self) -> List[str]:
        return [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", str(self.workers),
            "--timeout-keep-alive", str(self.timeout),
            "--log-level", "critical"
        ]

@dataclass
class RequestResult:
    request_id: int
    status_code: int
    response_time: float
    success: bool
    error_message: str = None

@dataclass
class LoadTestResult:
    worker_config: WorkerConfig
    total_requests: int
    concurrent_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    total_duration: float
    cpu_usage_percent: float
    memory_usage_mb: float

class UvicornBenchmarker:
    def __init__(self, test_pdf_path: str = None, output_dir: str = "evaluation/uvicorn_results"):
        # Auto-detect test PDF if not provided
        if test_pdf_path is None:
            pdf_dir = Path("evaluation/test_pdfs")
            if not pdf_dir.exists():
                raise FileNotFoundError("Folder 'evaluation/test_pdfs' not found! Create it and add at least one PDF.")
            
            pdf_files = list(pdf_dir.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in {pdf_dir}! Add sample.pdf or any PDF there.")
            
            test_pdf_path = str(pdf_files[0])
            print(f"Auto-detected test PDF → {test_pdf_path}")

        self.test_pdf_path = Path(test_pdf_path)
        if not self.test_pdf_path.exists():
            raise FileNotFoundError(f"Test PDF not found: {self.test_pdf_path}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.token = None

    def start_server(self, config: WorkerConfig) -> bool:
        cmd = config.to_cmd()
        print(f"\nStarting Uvicorn with {config.workers} worker(s)...")
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for _ in range(40):
                time.sleep(1.5)
                try:
                    r = requests.get("http://localhost:8000/api/v1/health", timeout=3)
                    if r.status_code == 200:
                        print("Server ready!")
                        return True
                except:
                    pass
            print("Server failed to start in time")
            self.stop_server()
            return False
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False

    def stop_server(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except:
                self.process.kill()
            self.process = None

    def get_token(self) -> str | None:
        if self.token:
            return self.token
        try:
            r = requests.post("http://localhost:8000/api/v1/auth/login",
                            json={"username": "admin", "password": "changeme123"}, timeout=10)
            r.raise_for_status()
            self.token = r.json()["access_token"]
            return self.token
        except Exception as e:
            print(f"Login failed: {e}")
            return None

    def send_request(self, req_id: int, pdf_path: Path, token: str) -> RequestResult:
            start = time.time()
            try:
                with open(pdf_path, "rb") as f:
                    files = {"pdf": f}
                    data = {
                        "model": "FFDNet-L",
                        "confidence": "0.3",
                        "device": "cuda:0",   # ← THIS WAS MISSING
                        # "fast": "true",       # ← THIS WAS MISSING (quantized ONNX = faster)
                        "keep_existing": "false",
                        "use_signature_fields": "false",
                        "multiline": "false"
                    }
                    headers = {"Authorization": f"Bearer {token}"}

                    r = requests.post(
                        "http://localhost:8000/api/v1/pdf/make-fillable",
                        files=files,
                        data=data,
                        headers=headers,
                        timeout=600
                    )

                elapsed = time.time() - start
                success = r.status_code == 200
                if not success:
                    print(f"Request {req_id} failed: {r.status_code} {r.text[:200]}")
                return RequestResult(req_id, r.status_code, elapsed, success)

            except Exception as e:
                return RequestResult(req_id, 0, time.time() - start, False, str(e))

    def run_load_test(self, pdf_path: Path, n_requests: int, concurrency: int, token: str):
        print(f"Running {n_requests} requests with {concurrency} concurrency...")
        results = []
        completed = [0]

        def worker(ids):
            for i in ids:
                results.append(self.send_request(i, pdf_path, token))
                completed[0] += 1
                if completed[0] % 10 == 0 or completed[0] == n_requests:
                    print(f"   Progress: {completed[0]}/{n_requests}")

        threads = []
        chunk = max(1, n_requests // concurrency)
        for i in range(0, n_requests, chunk):
            end = min(i + chunk, n_requests)
            t = threading.Thread(target=worker, args=(list(range(i, end)),))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return results

    def measure_resources(self, duration=5.0):
        if not self.process:
            return 0.0, 0.0
        try:
            proc = psutil.Process(self.process.pid)
            children = proc.children(recursive=True)
            all_p = [proc] + children
            cpu_vals, mem_vals = [], []
            end = time.time() + duration
            while time.time() < end:
                cpu = sum(p.cpu_percent() for p in all_p if p.status() == psutil.STATUS_RUNNING)
                mem = sum(p.memory_info().rss for p in all_p if p.status() == psutil.STATUS_RUNNING) / (1024**2)
                cpu_vals.append(cpu)
                mem_vals.append(mem)
                time.sleep(0.5)
            return (statistics.mean(cpu_vals) if cpu_vals else 0,
                    statistics.mean(mem_vals) if mem_vals else 0)
        except:
            return 0.0, 0.0

    def benchmark(self, config: WorkerConfig, n_requests=60, concurrency=15) -> LoadTestResult:
        print(f"\n{'='*80}")
        print(f" BENCHMARK: {config.workers} UVICORN WORKER(S) ")
        print(f"{'='*80}")

        if not self.start_server(config):
            raise RuntimeError("Failed to start server")

        try:
            token = self.get_token()
            if not token:
                raise RuntimeError("Authentication failed")

            # Warmup
            self.send_request(-1, self.test_pdf_path, token)
            time.sleep(3)

            start_time = time.time()
            results = self.run_load_test(self.test_pdf_path, n_requests, concurrency, token)
            duration = time.time() - start_time

            cpu, mem = self.measure_resources(6)

            successful_times = sorted([r.response_time for r in results if r.success])
            if not successful_times:
                raise RuntimeError("All requests failed!")

            result = LoadTestResult(
                worker_config=config,
                total_requests=n_requests,
                concurrent_requests=concurrency,
                successful_requests=len(successful_times),
                failed_requests=n_requests - len(successful_times),
                avg_response_time=round(statistics.mean(successful_times), 3),
                min_response_time=round(successful_times[0], 3),
                max_response_time=round(successful_times[-1], 3),
                median_response_time=round(statistics.median(successful_times), 3),
                p95_response_time=round(successful_times[int(len(successful_times) * 0.95)], 3),
                p99_response_time=round(successful_times[int(len(successful_times) * 0.99)], 3) if len(successful_times) > 10 else 0,
                requests_per_second=round(n_requests / duration, 2),
                total_duration=round(duration, 2),
                cpu_usage_percent=round(cpu, 1),
                memory_usage_mb=round(mem, 1)
            )

            print(f"RESULT → {result.requests_per_second} req/s | Avg {result.avg_response_time}s | P95 {result.p95_response_time}s | RAM {result.memory_usage_mb} MB")
            return result

        finally:
            self.stop_server()
            time.sleep(3)

    def run_all(self, worker_counts=None, n_requests=60, concurrency=15):
        if worker_counts is None:
            worker_counts = [1, 2, 4, 6, 8]

        results = []
        for workers in worker_counts:
            try:
                res = self.benchmark(WorkerConfig(workers), n_requests, concurrency)
                results.append(res)
            except Exception as e:
                print(f"Failed with {workers} workers: {e}")

        self.save_results(results)
        self.print_summary(results)

    def save_results(self, results: List[LoadTestResult]):
        file = self.output_dir / f"uvicorn_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
        data = {
            "generated_at": datetime.now().isoformat(),
            "test_pdf": str(self.test_pdf_path),
            "results": [{**asdict(r), "worker_config": asdict(r.worker_config)} for r in results]
        }
        with open(file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved → {file}")

    def print_summary(self, results: List[LoadTestResult]):
        print(f"\n{'='*100}")
        print(" FINAL UVICORN SCALING RESULTS ".center(100))
        print(f"{'='*100}")
        print(f"{'Workers':<8} {'RPS':<10} {'Avg (s)':<12} {'P95 (s)':<12} {'CPU %':<10} {'RAM (MB)':<12}")
        print("-" * 100)
        for r in results:
            print(f"{r.worker_config.workers:<8} {r.requests_per_second:<10} {r.avg_response_time:<12.2f} {r.p95_response_time:<12.2f} {r.cpu_usage_percent:<10.1f} {r.memory_usage_mb:<12.0f}")
        print(f"{'='*100}")
        if results:
            best = max(results, key=lambda x: x.requests_per_second)
            print(f"Best throughput: {best.worker_config.workers} workers → {best.requests_per_second} req/s")

# ————————————————————————————————
# AUTO-RUN WHEN EXECUTED DIRECTLY
# ————————————————————————————————
if __name__ == "__main__":
    benchmarker = UvicornBenchmarker()  # ← Fully automatic!
    benchmarker.run_all(
        worker_counts=[1, 2, 4, 6, 8],
        n_requests=60,
        concurrency=15
    )