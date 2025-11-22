import json
import time
import subprocess
import threading
import psutil
import requests
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
import sys
import os

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

class GunicornBenchmarker:
    def __init__(self, output_dir: str = "evaluation/gunicorn_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.server_process = None

    def start_server(self, workers: int):
        print(f"\nStarting Gunicorn with {workers} worker(s)...")
        
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["ULTRALYTICS_HUB"] = "off"
        
        # GUNICORN COMMAND
        # -k uvicorn.workers.UvicornWorker : Tells Gunicorn to use Uvicorn for ASGI
        # --bind : Specifies the host and port
        # --chdir : Ensures we are running from the correct directory context if needed
        cmd = [
            "gunicorn", 
            "app.main:app",
            "-k", "uvicorn.workers.UvicornWorker", 
            "--bind", "0.0.0.0:8000",
            "--workers", str(workers),
            "--log-level", "critical", # Keep it quiet
            "--timeout", "120" # Increase timeout for slow heavy loads
        ]
        
        # Start the process
        self.server_process = subprocess.Popen(cmd, env=env)
        
        # Wait for server to be ready
        print("Waiting for Gunicorn to boot...")
        for _ in range(45): # Gunicorn might take a split second longer to fork workers
            try:
                r = requests.get("http://localhost:8000/api/v1/health", timeout=2)
                if r.status_code == 200:
                    print("Server ready!")
                    time.sleep(3)  # Give workers a moment to fully initialize
                    return True
            except:
                time.sleep(1)
        
        print("Server failed to start!")
        self.stop_server()
        return False

    def stop_server(self):
        if self.server_process:
            print("Stopping Gunicorn...")
            # Gunicorn handles SIGTERM by gracefully shutting down workers
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except:
                self.server_process.kill()
            self.server_process = None
            time.sleep(2) # Allow OS to reclaim port 8000

    def get_token(self) -> str:
        # Retry logic for login
        for _ in range(5):
            try:
                r = requests.post("http://localhost:8000/api/v1/auth/login",
                                json={"username": "admin", "password": "changeme123"}, timeout=5)
                if r.status_code == 200:
                    return r.json()["access_token"]
            except:
                time.sleep(1)
        raise Exception("Could not login to get token")

    def send_request(self, pdf_path: str, token: str) -> float:
        start = time.time()
        try:
            with open(pdf_path, "rb") as f:
                files = {"pdf": f}
                # Explicitly requesting GPU (cuda:0)
                data = {"model": "FFDNet-L", "confidence": "0.3", "device": "cuda:0"}
                headers = {"Authorization": f"Bearer {token}"}
                r = requests.post("http://localhost:8000/api/v1/pdf/make-fillable",
                                  files=files, data=data, headers=headers, timeout=300)
            return time.time() - start if r.status_code == 200 else -1
        except Exception as e:
            return -1

    def run_test(self, workers: int, pdf_path: str, requests_count: int = 50, concurrent: int = 10) -> PerformanceResult:
        if not self.start_server(workers):
            raise Exception("Server failed")

        try:
            token = self.get_token()
            results = []
            lock = threading.Lock()
            requests_done = 0

            def worker():
                nonlocal requests_done
                while True:
                    with lock:
                        if len(results) >= requests_count:
                            break
                    
                    rt = self.send_request(pdf_path, token)
                    
                    with lock:
                        if len(results) < requests_count:
                            results.append(rt)
                            requests_done += 1
                            if requests_done % 5 == 0:
                                print(f"  Progress: {requests_done}/{requests_count}", end='\r')

            threads = [threading.Thread(target=worker) for _ in range(concurrent)]
            for t in threads: t.start()
            
            start_time = time.time()
            for t in threads: t.join()
            duration = time.time() - start_time
            print() 

            successful = [r for r in results if r > 0]
            times = sorted(successful)
            rps = len(successful) / duration if duration > 0 else 0

            # Resource Monitoring (captures Gunicorn Master + All Workers)
            proc = psutil.Process(self.server_process.pid)
            try:
                children = proc.children(recursive=True)
                all_procs = [proc] + children
                cpu = sum([p.cpu_percent(interval=0.1) for p in all_procs])
                mem = sum([p.memory_info().rss for p in all_procs]) / (1024 * 1024)
            except:
                cpu, mem = 0.0, 0.0

            return PerformanceResult(
                test_config=f"{workers}_workers_gunicorn",
                num_workers=workers,
                total_requests=requests_count,
                concurrent_requests=concurrent,
                successful_requests=len(successful),
                failed_requests=requests_count - len(successful),
                avg_response_time=round(statistics.mean(times), 3) if times else 0,
                min_response_time=round(min(times), 3) if times else 0,
                max_response_time=round(max(times), 3) if times else 0,
                median_response_time=round(statistics.median(times), 3) if times else 0,
                p95_response_time=round(times[int(len(times)*0.95)], 3) if times else 0,
                p99_response_time=round(times[int(len(times)*0.99)], 3) if times else 0,
                requests_per_second=round(rps, 2),
                total_duration=round(duration, 2),
                cpu_usage_percent=cpu,
                memory_usage_mb=mem
            )
        finally:
            self.stop_server()

    def run_all(self, worker_list=[1,2,4,8], pdf="evaluation/test_pdfs/sample.pdf", requests=50, concurrent=10):
        all_results = []
        print(f"Starting Gunicorn Benchmark on PDF: {pdf}")
        
        for w in worker_list:
            print(f"\n{'='*60}")
            print(f" TESTING GUNICORN WITH {w} WORKER(S)")
            print(f"{'='*60}")
            try:
                result = self.run_test(w, pdf, requests, concurrent)
                all_results.append(result)
                print(f" → {result.requests_per_second} req/s | Avg: {result.avg_response_time}s | P95: {result.p95_response_time}s")
            except Exception as e:
                print(f"Test failed for {w} workers: {e}")
                self.stop_server()
                
        self.save_results(all_results)

    def save_results(self, results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"gunicorn_performance_{timestamp}.json"
        results_dict = [asdict(r) for r in results]
        
        with open(path, "w") as f:
            json.dump({"results": results_dict}, f, indent=2)
        print(f"\nResults saved: {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="evaluation/test_pdfs/sample.pdf")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--workers", nargs="+", type=int, default=[1,2,4,8])
    args = parser.parse_args()

    tester = GunicornBenchmarker()
    tester.run_all(args.workers, args.pdf, args.requests, args.concurrent)