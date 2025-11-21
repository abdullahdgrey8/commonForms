# evaluation/gunicorn_benchmarker.py
"""
Benchmark Gunicorn performance with different worker configurations
"""
import json
import time
import subprocess
import signal
import requests
import psutil
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import statistics

@dataclass
class WorkerConfig:
    """Gunicorn worker configuration"""
    workers: int
    worker_class: str = "uvicorn.workers.UvicornWorker"
    threads: int = 1
    timeout: int = 300
    
    def to_args(self) -> List[str]:
        """Convert to gunicorn command arguments"""
        return [
            '-w', str(self.workers),
            '-k', self.worker_class,
            '--threads', str(self.threads),
            '--timeout', str(self.timeout),
            '--bind', '0.0.0.0:8000',
            '--access-logfile', '-',
            '--error-logfile', '-'
        ]


@dataclass
class RequestResult:
    """Result of a single request"""
    request_id: int
    status_code: int
    response_time: float
    success: bool
    error_message: str = None


@dataclass
class LoadTestResult:
    """Result of a load test"""
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


class GunicornBenchmarker:
    """Benchmark Gunicorn performance"""
    
    def __init__(
        self,
        app_module: str = "app.main:app",
        test_pdf_path: str = None,
        output_dir: str = "evaluation/gunicorn_results"
    ):
        self.app_module = app_module
        self.test_pdf_path = test_pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.token = None
        
    def start_gunicorn(self, config: WorkerConfig) -> bool:
        """Start Gunicorn server with specified configuration"""
        cmd = ['gunicorn'] + config.to_args() + [self.app_module]
        
        print(f"\nStarting Gunicorn with {config.workers} workers...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            max_wait = 30
            start_time = time.time()
            while time.time() - start_time < max_wait:
                try:
                    response = requests.get('http://localhost:8000/api/v1/health', timeout=1)
                    if response.status_code == 200:
                        print("✓ Server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
            
            print("✗ Server failed to start within timeout")
            self.stop_gunicorn()
            return False
            
        except Exception as e:
            print(f"✗ Error starting server: {e}")
            return False
    
    def stop_gunicorn(self):
        """Stop Gunicorn server"""
        if self.process:
            print("Stopping Gunicorn...")
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
    
    def get_auth_token(self, username: str = "admin", password: str = "changeme123") -> str:
        """Get authentication token"""
        try:
            response = requests.post(
                'http://localhost:8000/api/v1/auth/login',
                json={'username': username, 'password': password}
            )
            if response.status_code == 200:
                return response.json()['access_token']
            else:
                raise Exception(f"Login failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Failed to get token: {e}")
            return None
    
    def send_request(self, request_id: int, pdf_path: str, token: str) -> RequestResult:
        """Send a single request to the API"""
        start_time = time.time()
        
        try:
            with open(pdf_path, 'rb') as f:
                files = {'pdf': f}
                data = {'model': 'FFDNet-L', 'confidence': 0.3}
                headers = {'Authorization': f'Bearer {token}'}
                
                response = requests.post(
                    'http://localhost:8000/api/v1/pdf/make-fillable',
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=300
                )
            
            response_time = time.time() - start_time
            
            return RequestResult(
                request_id=request_id,
                status_code=response.status_code,
                response_time=response_time,
                success=response.status_code == 200
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return RequestResult(
                request_id=request_id,
                status_code=0,
                response_time=response_time,
                success=False,
                error_message=str(e)
            )
    
    def run_load_test(
        self,
        pdf_path: str,
        num_requests: int,
        concurrent: int,
        token: str
    ) -> List[RequestResult]:
        """Run load test with specified concurrency"""
        print(f"\nRunning load test: {num_requests} requests, {concurrent} concurrent")
        
        results = []
        completed = 0
        
        def worker(request_ids: List[int]):
            nonlocal completed
            for req_id in request_ids:
                result = self.send_request(req_id, pdf_path, token)
                results.append(result)
                completed += 1
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{num_requests}")
        
        # Split requests among threads
        request_ids = list(range(num_requests))
        chunk_size = (num_requests + concurrent - 1) // concurrent
        chunks = [request_ids[i:i+chunk_size] for i in range(0, num_requests, chunk_size)]
        
        # Start concurrent threads
        threads = []
        for chunk in chunks:
            thread = threading.Thread(target=worker, args=(chunk,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        return results
    
    def measure_system_resources(self, duration: float = 5.0) -> Tuple[float, float]:
        """Measure average CPU and memory usage"""
        if not self.process:
            return 0.0, 0.0
        
        try:
            parent = psutil.Process(self.process.pid)
            children = parent.children(recursive=True)
            processes = [parent] + children
            
            cpu_samples = []
            mem_samples = []
            
            start_time = time.time()
            while time.time() - start_time < duration:
                total_cpu = sum(p.cpu_percent() for p in processes if p.is_running())
                total_mem = sum(p.memory_info().rss for p in processes if p.is_running())
                
                cpu_samples.append(total_cpu)
                mem_samples.append(total_mem / (1024 * 1024))  # MB
                
                time.sleep(0.5)
            
            avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0
            avg_mem = statistics.mean(mem_samples) if mem_samples else 0
            
            return avg_cpu, avg_mem
            
        except Exception as e:
            print(f"Warning: Could not measure system resources: {e}")
            return 0.0, 0.0
    
    def benchmark_configuration(
        self,
        config: WorkerConfig,
        num_requests: int,
        concurrent: int,
        pdf_path: str
    ) -> LoadTestResult:
        """Benchmark a single worker configuration"""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {config.workers} workers, {config.threads} threads")
        print(f"{'='*60}")
        
        # Start server
        if not self.start_gunicorn(config):
            raise Exception("Failed to start Gunicorn")
        
        try:
            # Get auth token
            token = self.get_auth_token()
            if not token:
                raise Exception("Failed to get authentication token")
            
            # Warm up
            print("Warming up...")
            self.send_request(0, pdf_path, token)
            time.sleep(2)
            
            # Measure baseline resources
            print("Measuring baseline resources...")
            baseline_cpu, baseline_mem = self.measure_system_resources(duration=3)
            
            # Run load test
            start_time = time.time()
            results = self.run_load_test(pdf_path, num_requests, concurrent, token)
            total_duration = time.time() - start_time
            
            # Measure resources during load
            print("Measuring resources under load...")
            load_cpu, load_mem = self.measure_system_resources(duration=5)
            
            # Calculate statistics
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            if not successful:
                raise Exception("All requests failed")
            
            response_times = [r.response_time for r in successful]
            response_times.sort()
            
            avg_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            p95_time = response_times[int(len(response_times) * 0.95)]
            p99_time = response_times[int(len(response_times) * 0.99)]
            
            rps = num_requests / total_duration
            
            result = LoadTestResult(
                worker_config=config,
                total_requests=num_requests,
                concurrent_requests=concurrent,
                successful_requests=len(successful),
                failed_requests=len(failed),
                avg_response_time=round(avg_time, 3),
                min_response_time=round(min(response_times), 3),
                max_response_time=round(max(response_times), 3),
                median_response_time=round(median_time, 3),
                p95_response_time=round(p95_time, 3),
                p99_response_time=round(p99_time, 3),
                requests_per_second=round(rps, 2),
                total_duration=round(total_duration, 2),
                cpu_usage_percent=round(load_cpu, 1),
                memory_usage_mb=round(load_mem, 1)
            )
            
            print(f"\n✓ Benchmark complete:")
            print(f"  Successful: {len(successful)}/{num_requests}")
            print(f"  Avg response time: {avg_time:.2f}s")
            print(f"  Median: {median_time:.2f}s, P95: {p95_time:.2f}s")
            print(f"  Throughput: {rps:.2f} req/s")
            print(f"  CPU: {load_cpu:.1f}%, Memory: {load_mem:.1f} MB")
            
            return result
            
        finally:
            self.stop_gunicorn()
            time.sleep(3)  # Wait between tests
    
    def run_worker_comparison(
        self,
        worker_counts: List[int],
        num_requests: int = 50,
        concurrent: int = 10,
        pdf_path: str = None
    ) -> List[LoadTestResult]:
        """Compare performance across different worker counts"""
        if pdf_path is None:
            pdf_path = self.test_pdf_path
        
        if not pdf_path or not Path(pdf_path).exists():
            raise Exception("Test PDF path not provided or doesn't exist")
        
        results = []
        
        for worker_count in worker_counts:
            config = WorkerConfig(workers=worker_count)
            try:
                result = self.benchmark_configuration(
                    config, num_requests, concurrent, pdf_path
                )
                results.append(result)
            except Exception as e:
                print(f"✗ Benchmark failed for {worker_count} workers: {e}")
        
        return results
    
    def save_results(self, results: List[LoadTestResult], output_file: str = None):
        """Save benchmark results"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"gunicorn_benchmark_{timestamp}.json"
        else:
            output_file = Path(output_file)
        
        data = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'results': []
        }
        
        for result in results:
            result_dict = asdict(result)
            result_dict['worker_config'] = asdict(result.worker_config)
            data['results'].append(result_dict)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"{'Workers':<10} {'RPS':<10} {'Avg Time':<12} {'P95 Time':<12} {'CPU %':<10} {'Memory MB':<12}")
        print(f"{'-'*60}")
        
        for result in results:
            print(
                f"{result.worker_config.workers:<10} "
                f"{result.requests_per_second:<10.2f} "
                f"{result.avg_response_time:<12.2f} "
                f"{result.p95_response_time:<12.2f} "
                f"{result.cpu_usage_percent:<10.1f} "
                f"{result.memory_usage_mb:<12.1f}"
            )
        
        print(f"\nResults saved to: {output_file}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    benchmarker = GunicornBenchmarker(
        app_module="app.main:app",
        test_pdf_path="test.pdf"
    )
    
    # Test different worker counts
    worker_counts = [1, 2, 4, 8]
    
    results = benchmarker.run_worker_comparison(
        worker_counts=worker_counts,
        num_requests=50,
        concurrent=10
    )
    
    benchmarker.save_results(results)
