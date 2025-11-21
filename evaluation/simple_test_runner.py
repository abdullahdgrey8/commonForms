# evaluation/simple_test.py
"""
Simple test runner for quick evaluation
"""
import sys
import requests
from pathlib import Path
from accuracy_eval import AccuracyEvaluator
from gunicorn_benchmarker import GunicornBenchmarker

def test_accuracy(api_url: str, token: str, pdf_dir: str):
    """Run simple accuracy test"""
    print("\n" + "="*60)
    print("ACCURACY TEST")
    print("="*60)
    
    evaluator = AccuracyEvaluator(api_base_url=api_url, token=token)
    
    # Test with FFDNet-L at confidence 0.3
    metrics = evaluator.evaluate_dataset(
        pdf_dir=pdf_dir,
        model="FFDNet-L",
        confidence=0.3
    )
    
    print(f"\n✅ Results:")
    print(f"   PDFs tested: {metrics.total_pdfs}")
    print(f"   Successful: {metrics.successful_pdfs}")
    print(f"   Avg fields detected: {metrics.avg_fields_per_pdf}")
    print(f"   Avg processing time: {metrics.avg_processing_time:.2f}s")
    
    if metrics.field_type_distribution:
        print(f"\n   Field types:")
        for field_type, count in metrics.field_type_distribution.items():
            print(f"     - {field_type}: {count}")
    
    evaluator.save_results(metrics, "evaluation/quick_test_accuracy.json")
    print(f"\n   Results saved to: evaluation/quick_test_accuracy.json")
    
    return metrics


def test_performance(pdf_path: str):
    """Run simple performance test"""
    print("\n" + "="*60)
    print("PERFORMANCE TEST")
    print("="*60)
    
    benchmarker = GunicornBenchmarker(
        app_module="app.main:app",
        test_pdf_path=pdf_path
    )
    
    # Test with 1 and 2 workers
    results = benchmarker.run_worker_comparison(
        worker_counts=[1, 2],
        num_requests=10,
        concurrent=2
    )
    
    print(f"\n✅ Results:")
    for result in results:
        print(f"\n   {result.worker_config.workers} worker(s):")
        print(f"     - Throughput: {result.requests_per_second:.2f} req/s")
        print(f"     - Avg latency: {result.avg_response_time:.2f}s")
        print(f"     - P95 latency: {result.p95_response_time:.2f}s")
        print(f"     - CPU usage: {result.cpu_usage_percent:.1f}%")
        print(f"     - Memory: {result.memory_usage_mb:.1f} MB")
    
    benchmarker.save_results(results, "evaluation/quick_test_performance.json")
    print(f"\n   Results saved to: evaluation/quick_test_performance.json")
    
    return results


def main():
    """Main test runner"""
    print("\n" + "="*60)
    print("CommonForms Quick Test Runner")
    print("="*60)
    
    # Configuration
    api_url = input("\nAPI URL [http://localhost:8000]: ").strip() or "http://localhost:8000"
    username = input("Username [admin]: ").strip() or "admin"
    password = input("Password [changeme123]: ").strip() or "changeme123"
    
    # Check API
    print(f"\nChecking API at {api_url}...")
    try:
        response = requests.get(f"{api_url}/api/v1/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ API returned status {response.status_code}")
            sys.exit(1)
        print("✅ API is running")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        sys.exit(1)
    
    # Get token
    print(f"\nAuthenticating as {username}...")
    try:
        response = requests.post(
            f"{api_url}/api/v1/auth/login",
            json={"username": username, "password": password}
        )
        if response.status_code != 200:
            print(f"❌ Authentication failed: {response.status_code}")
            sys.exit(1)
        token = response.json()["access_token"]
        print("✅ Authentication successful")
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        sys.exit(1)
    
    # Test selection
    print("\nWhat would you like to test?")
    print("1. Accuracy (requires PDF directory)")
    print("2. Performance (requires single PDF)")
    print("3. Both")
    
    choice = input("\nChoice [1-3]: ").strip()
    
    if choice in ["1", "3"]:
        pdf_dir = input("\nPDF directory [evaluation/test_pdfs]: ").strip() or "evaluation/test_pdfs"
        if not Path(pdf_dir).exists():
            print(f"❌ Directory not found: {pdf_dir}")
            sys.exit(1)
        
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_dir}")
            sys.exit(1)
        
        print(f"✅ Found {len(pdf_files)} PDF file(s)")
        test_accuracy(api_url, token, pdf_dir)
    
    if choice in ["2", "3"]:
        if choice == "3":
            # Use first PDF from directory
            pdf_path = str(pdf_files[0])
            print(f"\nUsing {pdf_path} for performance test")
        else:
            pdf_path = input("\nTest PDF path: ").strip()
            if not Path(pdf_path).exists():
                print(f"❌ File not found: {pdf_path}")
                sys.exit(1)
        
        test_performance(pdf_path)
    
    print("\n" + "="*60)
    print("✅ Tests Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  - Review results in evaluation/ directory")
    print("  - Run full evaluation: python evaluation/run_full_evaluation.py --help")
    print("  - Use quick_start.sh for guided testing")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
