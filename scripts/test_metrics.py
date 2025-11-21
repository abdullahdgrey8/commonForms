# scripts/test_metrics.py
"""
Test script for metrics and benchmarking features

Usage:
    python scripts/test_metrics.py --pdf test.pdf
"""

import requests
import argparse
import time
from pathlib import Path


class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token = None
    
    def login(self, username: str = "admin", password: str = "changeme123"):
        """Login and get token"""
        print(f"🔐 Logging in as {username}...")
        response = requests.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": username, "password": password}
        )
        response.raise_for_status()
        self.token = response.json()["access_token"]
        print(f"✅ Login successful")
        return self.token
    
    def test_basic_processing(self, pdf_path: str):
        """Test basic PDF processing without metrics"""
        print(f"\n📄 Testing basic processing: {pdf_path}")
        
        files = {"pdf": open(pdf_path, "rb")}
        data = {
            "model": "FFDNet-L",
            "device": "cpu",
            "confidence": 0.3,
            "track_metrics": False
        }
        headers = {"Authorization": f"Bearer {self.token}"}
        
        start = time.time()
        response = requests.post(
            f"{self.base_url}/api/v1/pdf/make-fillable",
            files=files,
            data=data,
            headers=headers
        )
        elapsed = time.time() - start
        
        response.raise_for_status()
        
        print(f"✅ Processing completed in {elapsed:.2f}s")
        print(f"   PDF size: {len(response.content)} bytes")
        
        # Save output
        output_path = Path(pdf_path).stem + "_fillable_basic.pdf"
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"   Saved to: {output_path}")
    
    def test_metrics_processing(self, pdf_path: str, device: str = "cpu"):
        """Test PDF processing with metrics"""
        print(f"\n📊 Testing with metrics tracking: {pdf_path}")
        
        files = {"pdf": open(pdf_path, "rb")}
        data = {
            "model": "FFDNet-L",
            "device": device,
            "confidence": 0.3,
            "track_metrics": True
        }
        headers = {"Authorization": f"Bearer {self.token}"}
        
        start = time.time()
        response = requests.post(
            f"{self.base_url}/api/v1/pdf/make-fillable",
            files=files,
            data=data,
            headers=headers
        )
        elapsed = time.time() - start
        
        response.raise_for_status()
        
        # Extract metrics from headers
        request_id = response.headers.get('X-Request-ID', 'N/A')
        processing_time = response.headers.get('X-Processing-Time', 'N/A')
        input_dpi = response.headers.get('X-Input-DPI', 'N/A')
        output_dpi = response.headers.get('X-Output-DPI', 'N/A')
        gpu_memory = response.headers.get('X-GPU-Peak-Memory-MB', 'N/A')
        
        print(f"✅ Processing completed in {elapsed:.2f}s")
        print(f"   Request ID: {request_id}")
        print(f"   Processing Time: {processing_time}s")
        print(f"   Input DPI: {input_dpi}")
        print(f"   Output DPI: {output_dpi}")
        print(f"   GPU Peak Memory: {gpu_memory} MB")
        
        # Save output
        output_path = Path(pdf_path).stem + "_fillable_metrics.pdf"
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"   Saved to: {output_path}")
        
        return request_id
    
    def test_concurrent_benchmark(self, pdf_path: str, num_concurrent: int = 5):
        """Test concurrent benchmark"""
        print(f"\n🏃 Testing concurrent benchmark: {num_concurrent} requests")
        
        files = {"pdf": open(pdf_path, "rb")}
        data = {
            "num_concurrent": num_concurrent,
            "model": "FFDNet-L",
            "device": "cpu",
            "confidence": 0.3
        }
        headers = {"Authorization": f"Bearer {self.token}"}
        
        start = time.time()
        response = requests.post(
            f"{self.base_url}/api/v1/pdf/benchmark",
            files=files,
            data=data,
            headers=headers
        )
        elapsed = time.time() - start
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Benchmark completed in {elapsed:.2f}s")
        print(f"   Total Requests: {result['total_requests']}")
        print(f"   Successful: {result['successful_requests']}")
        print(f"   Failed: {result['failed_requests']}")
        print(f"   Avg Time: {result['average_processing_time']:.3f}s")
        print(f"   Min Time: {result['min_processing_time']:.3f}s")
        print(f"   Max Time: {result['max_processing_time']:.3f}s")
        print(f"   Throughput: {result['requests_per_second']:.3f} req/s")
    
    def test_sequential_benchmark(self, pdf_path: str, num_requests: int = 3):
        """Test sequential benchmark"""
        print(f"\n🔄 Testing sequential benchmark: {num_requests} requests")
        
        files = {"pdf": open(pdf_path, "rb")}
        data = {
            "num_requests": num_requests,
            "model": "FFDNet-L",
            "device": "cpu",
            "confidence": 0.3
        }
        headers = {"Authorization": f"Bearer {self.token}"}
        
        start = time.time()
        response = requests.post(
            f"{self.base_url}/api/v1/pdf/benchmark/sequential",
            files=files,
            data=data,
            headers=headers
        )
        elapsed = time.time() - start
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Sequential benchmark completed in {elapsed:.2f}s")
        print(f"   Total Requests: {result['total_requests']}")
        print(f"   Successful: {result['successful_requests']}")
        print(f"   Avg Time: {result['average_processing_time']:.3f}s")
        print(f"   Min Time: {result['min_processing_time']:.3f}s")
        print(f"   Max Time: {result['max_processing_time']:.3f}s")
    
    def get_metrics_summary(self):
        """Get metrics summary"""
        print(f"\n📈 Getting metrics summary...")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(
            f"{self.base_url}/api/v1/pdf/metrics/summary",
            headers=headers
        )
        response.raise_for_status()
        
        summary = response.json()
        print(f"✅ Metrics Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
    
    def get_recent_metrics(self, limit: int = 10):
        """Get recent metrics"""
        print(f"\n📋 Getting recent metrics (limit={limit})...")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(
            f"{self.base_url}/api/v1/pdf/metrics/recent?limit={limit}",
            headers=headers
        )
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Retrieved {data['count']} metrics")
        
        if data['metrics']:
            print("\n   Recent requests:")
            for m in data['metrics'][:5]:  # Show first 5
                print(f"   - {m['request_id']}: {m['filename']} "
                      f"({m['total_processing_time']:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description='Test metrics features')
    parser.add_argument('--pdf', type=str, required=True, help='Path to test PDF')
    parser.add_argument('--base-url', type=str, default='http://localhost:8000',
                       help='API base URL')
    parser.add_argument('--username', type=str, default='admin',
                       help='Username for authentication')
    parser.add_argument('--password', type=str, default='changeme123',
                       help='Password for authentication')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda:0)')
    parser.add_argument('--skip-basic', action='store_true',
                       help='Skip basic processing test')
    parser.add_argument('--skip-benchmark', action='store_true',
                       help='Skip benchmark tests')
    parser.add_argument('--concurrent', type=int, default=5,
                       help='Number of concurrent requests for benchmark')
    
    args = parser.parse_args()
    
    # Verify PDF exists
    if not Path(args.pdf).exists():
        print(f"❌ PDF file not found: {args.pdf}")
        return
    
    print("="*60)
    print("🧪 COMMONFORMS API METRICS TEST SUITE")
    print("="*60)
    
    tester = APITester(args.base_url)
    
    try:
        # Login
        tester.login(args.username, args.password)
        
        # Test basic processing
        if not args.skip_basic:
            tester.test_basic_processing(args.pdf)
        
        # Test with metrics
        tester.test_metrics_processing(args.pdf, args.device)
        
        # Test benchmarks
        if not args.skip_benchmark:
            tester.test_sequential_benchmark(args.pdf, num_requests=3)
            tester.test_concurrent_benchmark(args.pdf, num_concurrent=args.concurrent)
        
        # Get metrics summary
        tester.get_metrics_summary()
        tester.get_recent_metrics(limit=10)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Test failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == '__main__':
    main()