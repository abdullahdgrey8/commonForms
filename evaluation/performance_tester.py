# evaluation/performance_tester.py
# Cross-platform version (Linux/macOS/WSL)

import json
import time
import subprocess
import threading
import psutil
import requests
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

@dataclass
class PerformanceResult:
    test_config: str
    num_workers: int
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
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0

class PerformanceTester:
    def __init__(self, output_dir: str = "evaluation/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.server_process = None

    def start_server(self, workers: int):
        print(f"\nStarting Uvicorn with {workers} worker(s)...")
        cmd = [
            "uvicorn", "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", str(workers),
            "--log-level", "critical"
        ]
        self.server_process = subprocess.Popen(cmd)
        
        # Wait for server to be ready
        for _ in range(30):
            try:
                r = requests.get("http://localhost:8000/api/v1/health", timeout=2)
                if r.status_code == 200:
                    print("Server ready!")
                    time.sleep(2)  # Warm up
                    return True
            except:
                time.sleep(1)
        print("Server failed to start!")
        return False

    def stop_server(self):
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except:
                self.server_process.kill()
            self.server_process = None
            time.sleep(3)

    def get_token(self) -> str:
        r = requests.post("http://localhost:8000/api/v1/auth/login",
                          json={"username": "admin", "password": "changeme123"})
        return r.json()["access_token"]

    def send_request(self, pdf_path: str, token: str) -> float:
        start = time.time()
        try:
            with open(pdf_path, "rb") as f:
                files = {"pdf": f}
                data = {"model": "FFDNet-L", "confidence": "0.3", "device": "cuda:0"}
                headers = {"Authorization": f"Bearer {token}"}
                r = requests.post("http://localhost:8000/api/v1/pdf/make-fillable",
                                  files=files, data=data, headers=headers, timeout=300)
            return time.time() - start if r.status_code == 200 else -1
        except:
            return -1

    def run_test(self, workers: int, pdf_path: str, requests: int = 50, concurrent: int = 10) -> PerformanceResult:
        if not self.start_server(workers):
            raise Exception("Server failed")

        token = self.get_token()
        results = []

        def worker():
            while len(results) < requests:
                rt = self.send_request(pdf_path, token)
                if rt > 0:
                    results.append(rt)

        threads = [threading.Thread(target=worker) for _ in range(concurrent)]
        for t in threads: t.start()
        
        start_time = time.time()
        for t in threads: t.join()
        duration = time.time() - start_time

        successful = [r for r in results if r > 0]
        times = sorted(successful)
        rps = len(successful) / duration if duration > 0 else 0

        return PerformanceResult(
            test_config=f"{workers}_workers",
            num_workers=workers,
            total_requests=requests,
            concurrent_requests=concurrent,
            successful_requests=len(successful),
            failed_requests=requests - len(successful),
            avg_response_time=round(statistics.mean(times), 3) if times else 0,
            min_response_time=round(min(times), 3) if times else 0,
            max_response_time=round(max(times), 3) if times else 0,
            median_response_time=round(statistics.median(times), 3) if times else 0,
            p95_response_time=round(times[int(len(times)*0.95)]) if times else 0,
            p99_response_time=round(times[int(len(times)*0.99)]) if times else 0,
            requests_per_second=round(rps, 2),
            total_duration=round(duration, 2)
        )

    def run_all(self, worker_list=[1,2,4,8], pdf="evaluation/test_pdfs/sample.pdf", requests=50, concurrent=10):
        all_results = []
        for w in worker_list:
            print(f"\n{'='*60}")
            print(f" TESTING {w} WORKER(S)")
            print(f"{'='*60}")
            try:
                result = self.run_test(w, pdf, requests, concurrent)
                all_results.append(result)
                print(f" → {result.requests_per_second} req/s | {result.avg_response_time}s avg")
            finally:
                self.stop_server()
        self.save_results(all_results)

    def save_results(self, results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"performance_{timestamp}.json"
        with open(path, "w") as f:
            json.dump({"results": [asdict(r) for r in results]}, f, indent=2)
        print(f"\nResults saved: {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="evaluation/test_pdfs/sample.pdf")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--workers", nargs="+", type=int, default=[1,2,4,8])
    args = parser.parse_args()

    tester = PerformanceTester()
    tester.run_all(args.workers, args.pdf, args.requests, args.concurrent)