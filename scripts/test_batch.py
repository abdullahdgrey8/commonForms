# scripts/test_batch.py
"""
Test script for batch processing

Usage:
    python scripts/test_batch.py --pdfs file1.pdf file2.pdf file3.pdf
    python scripts/test_batch.py --pdfs *.pdf --parallel
    python scripts/test_batch.py --pdfs *.pdf --device cuda:0
"""

import requests
import argparse
from pathlib import Path
import time


def login(base_url: str, username: str, password: str) -> str:
    """Login and get token"""
    print(f"🔐 Logging in as {username}...")
    response = requests.post(
        f"{base_url}/api/v1/auth/login",
        json={"username": username, "password": password}
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    print(f"✅ Login successful\n")
    return token


def test_batch_processing(
    base_url: str,
    token: str,
    pdf_files: list,
    device: str = "cpu",
    parallel: bool = True,
    track_metrics: bool = True
):
    """Test batch processing"""
    print(f"📦 Testing batch processing:")
    print(f"   Files: {len(pdf_files)}")
    print(f"   Device: {device}")
    print(f"   Parallel: {parallel}")
    print(f"   Track metrics: {track_metrics}\n")
    
    # Prepare files
    files = []
    for pdf_path in pdf_files:
        path = Path(pdf_path)
        if not path.exists():
            print(f"⚠️  File not found: {pdf_path}")
            continue
        files.append(('pdfs', (path.name, open(pdf_path, 'rb'), 'application/pdf')))
    
    if not files:
        print("❌ No valid PDF files found")
        return
    
    print(f"📤 Uploading {len(files)} files...")
    
    # Prepare request
    data = {
        'model': 'FFDNet-L',
        'device': device,
        'parallel': str(parallel).lower(),
        'track_metrics': str(track_metrics).lower(),
        'confidence': '0.3'
    }
    
    headers = {'Authorization': f'Bearer {token}'}
    
    # Send request
    start_time = time.time()
    try:
        response = requests.post(
            f'{base_url}/api/v1/pdf/make-fillable-batch',
            files=files,
            data=data,
            headers=headers,
            timeout=300  # 5 minutes timeout
        )
        response.raise_for_status()
        
        elapsed = time.time() - start_time
        result = response.json()
        
        print(f"\n{'='*60}")
        print(f"📊 BATCH PROCESSING RESULTS")
        print(f"{'='*60}")
        print(f"Batch ID: {result['batch_id']}")
        print(f"Total files: {result['total_files']}")
        print(f"Successful: {result['successful']}")
        print(f"Failed: {result['failed']}")
        print(f"Processing time: {result['total_time']:.2f}s")
        print(f"API call time: {elapsed:.2f}s")
        print(f"\nIndividual Results:")
        print(f"{'-'*60}")
        
        for file_result in result['results']:
            status = "✅" if file_result['success'] else "❌"
            print(f"{status} {file_result['filename']}")
            print(f"   Time: {file_result['processing_time']:.2f}s")
            if file_result['success']:
                print(f"   Output: {file_result['output_filename']}")
                print(f"   Request ID: {file_result['request_id']}")
            else:
                print(f"   Error: {file_result['error_message']}")
            print()
        
        # Calculate statistics
        if result['successful'] > 0:
            successful_times = [
                r['processing_time'] 
                for r in result['results'] 
                if r['success']
            ]
            avg_time = sum(successful_times) / len(successful_times)
            print(f"Average processing time: {avg_time:.2f}s per file")
            print(f"Throughput: {result['successful'] / result['total_time']:.2f} files/sec")
        
        print(f"{'='*60}\n")
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>5 minutes)")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Close all file handles
        for _, (_, file_obj, _) in files:
            file_obj.close()


def compare_parallel_vs_sequential(
    base_url: str,
    token: str,
    pdf_files: list,
    device: str = "cpu"
):
    """Compare parallel vs sequential processing"""
    print(f"\n🏁 COMPARING PARALLEL VS SEQUENTIAL PROCESSING\n")
    
    # Test sequential
    print("1️⃣  Sequential processing...")
    test_batch_processing(base_url, token, pdf_files, device, parallel=False, track_metrics=False)
    
    # Wait a bit
    time.sleep(2)
    
    # Test parallel
    print("\n2️⃣  Parallel processing...")
    test_batch_processing(base_url, token, pdf_files, device, parallel=True, track_metrics=False)


def main():
    parser = argparse.ArgumentParser(description='Test batch processing')
    parser.add_argument('--pdfs', nargs='+', required=True, help='PDF files to process')
    parser.add_argument('--base-url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--username', default='admin', help='Username')
    parser.add_argument('--password', default='changeme123', help='Password')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda:0)')
    parser.add_argument('--parallel', action='store_true', help='Process in parallel')
    parser.add_argument('--sequential', action='store_true', help='Process sequentially')
    parser.add_argument('--compare', action='store_true', help='Compare parallel vs sequential')
    parser.add_argument('--no-metrics', action='store_true', help='Disable metrics tracking')
    
    args = parser.parse_args()
    
    # Expand wildcards if needed
    pdf_files = []
    for pattern in args.pdfs:
        if '*' in pattern or '?' in pattern:
            from glob import glob
            pdf_files.extend(glob(pattern))
        else:
            pdf_files.append(pattern)
    
    # Remove duplicates
    pdf_files = list(set(pdf_files))
    
    # Filter to only PDFs
    pdf_files = [f for f in pdf_files if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files\n")
    
    # Login
    try:
        token = login(args.base_url, args.username, args.password)
        
        if args.compare:
            # Compare modes
            compare_parallel_vs_sequential(args.base_url, token, pdf_files, args.device)
        else:
            # Single test
            parallel = args.parallel or not args.sequential  # Default to parallel
            test_batch_processing(
                args.base_url,
                token,
                pdf_files,
                args.device,
                parallel=parallel,
                track_metrics=not args.no_metrics
            )
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == '__main__':
    main()