# # evaluation/run_evaluation_no_plots.py
# """
# Complete evaluation suite for CommonForms API (without matplotlib dependency)
# """
# import argparse
# import json
# from pathlib import Path
# from datetime import datetime
# import sys
# import os

# # Add parent directory to path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from accuracy_eval import AccuracyEvaluator
# from gunicorn_benchmarker import GunicornBenchmarker

# class EvaluationSuite:
#     """Complete evaluation suite (no plotting)"""
    
#     def __init__(self, output_dir: str = "evaluation/complete_results"):
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#     def run_accuracy_evaluation(
#         self,
#         api_url: str,
#         token: str,
#         test_pdf_dir: str,
#         models: list = None,
#         confidences: list = None
#     ):
#         """Run comprehensive accuracy evaluation"""
#         print("\n" + "="*70)
#         print(f"ACCURACY EVALUATION ON: {test_pdf_dir}")
#         print("="*70)
        
#         if models is None:
#             models = ["FFDNet-L", "FFDNet-S"]
#         if confidences is None:
#             confidences = [0.2, 0.3, 0.4]
        
#         evaluator = AccuracyEvaluator(
#             api_base_url=api_url,
#             token=token,
#             output_dir=str(self.output_dir / "accuracy")
#         )
        
#         all_results = []
        
#         for model in models:
#             for confidence in confidences:
#                 print(f"\n--- Testing {model} with confidence={confidence} ---")
                
#                 metrics = evaluator.evaluate_dataset(
#                     pdf_dir=test_pdf_dir,
#                     model=model,
#                     confidence=confidence
#                 )
                
#                 result = {
#                     'model': model,
#                     'confidence': confidence,
#                     'metrics': metrics
#                 }
#                 all_results.append(result)
                
#                 # Save individual result
#                 output_file = (
#                     self.output_dir / "accuracy" / 
#                     f"{model}_{confidence}_{self.timestamp}.json"
#                 )
#                 evaluator.save_results(metrics, str(output_file))
        
#         return all_results
    
#     # ==============================================================================
#     # NEW FUNCTIONALITY: Noise Robustness Evaluation
#     # ==============================================================================
#     def run_noise_robustness_evaluation(
#         self,
#         api_url: str,
#         token: str,
#         clean_pdf_dir: str,
#         blurry_pdf_dir: str,
#         salt_pepper_pdf_dir: str,
#         models: list = None,
#         confidences: list = None
#     ):
#         """Run accuracy evaluation on clean, blurry, and salt&pepper PDFs and compare."""
#         print("\n" + "="*70)
#         print("NOISE ROBUSTNESS EVALUATION")
#         print("="*70)
        
#         # 1. Run evaluation on all three PDF sets
#         print("\n--- Step 1: Evaluating Clean PDFs ---")
#         clean_results = self.run_accuracy_evaluation(
#             api_url, token, clean_pdf_dir, models, confidences
#         )
        
#         print("\n--- Step 2: Evaluating Blurry PDFs ---")
#         blurry_results = self.run_accuracy_evaluation(
#             api_url, token, blurry_pdf_dir, models, confidences
#         )

#         print("\n--- Step 3: Evaluating Salt & Pepper PDFs ---")
#         sp_results = self.run_accuracy_evaluation(
#             api_url, token, salt_pepper_pdf_dir, models, confidences
#         )
        
#         # 2. Generate the comparison report
#         self._create_noise_comparison_report(clean_results, blurry_results, sp_results)
        
#         return clean_results, blurry_results, sp_results

#     def _create_noise_comparison_report(self, clean_results, blurry_results, sp_results):
#         """Create a detailed report comparing clean vs. noisy results."""
#         report_file = self.output_dir / f"noise_robustness_report_{self.timestamp}.txt"
        
#         # Create lookup maps for easy access to metrics by (model, confidence)
#         clean_map = {(r['model'], r['confidence']): r['metrics'] for r in clean_results}
#         blurry_map = {(r['model'], r['confidence']): r['metrics'] for r in blurry_results}
#         sp_map = {(r['model'], r['confidence']): r['metrics'] for r in sp_results}

#         with open(report_file, 'w') as f:
#             f.write("="*100 + "\n")
#             f.write("NOISE ROBUSTNESS EVALUATION REPORT\n")
#             f.write("="*100 + "\n\n")
            
#             # Header for the comparison table
#             f.write(f"{'Model':<12} {'Conf':<8} {'Metric':<20} {'Clean':<12} {'Blur':<12} {'S&P':<12} {'Blur Acc':<12} {'S&P Acc':<12}\n")
#             f.write("-"*100 + "\n")
            
#             # Iterate through each configuration and compare
#             for key, clean_metrics in clean_map.items():
#                 model, conf = key
#                 blurry_metrics = blurry_map.get(key)
#                 sp_metrics = sp_map.get(key)

#                 if not blurry_metrics or not sp_metrics:
#                     continue

#                 # Compare Average Fields Per PDF
#                 clean_fields = clean_metrics.avg_fields_per_pdf
#                 blurry_fields = blurry_metrics.avg_fields_per_pdf
#                 sp_fields = sp_metrics.avg_fields_per_pdf
                
#                 blur_field_acc = (blurry_fields / clean_fields * 100) if clean_fields > 0 else 0
#                 sp_field_acc = (sp_fields / clean_fields * 100) if clean_fields > 0 else 0
                
#                 f.write(
#                     f"{model:<12} {conf:<8.1f} {'Avg Fields':<20} "
#                     f"{clean_fields:<12.2f} {blurry_fields:<12.2f} {sp_fields:<12.2f} "
#                     f"{blur_field_acc:<12.1f}% {sp_field_acc:<12.1f}%\n"
#                 )

#                 # Compare Detection Rate
#                 clean_det_rate = clean_metrics.avg_detection_rate or 0
#                 blurry_det_rate = blurry_metrics.avg_detection_rate or 0
#                 sp_det_rate = sp_metrics.avg_detection_rate or 0

#                 blur_det_acc = (blurry_det_rate / clean_det_rate * 100) if clean_det_rate > 0 else 0
#                 sp_det_acc = (sp_det_rate / clean_det_rate * 100) if clean_det_rate > 0 else 0
                
#                 f.write(
#                     f"{model:<12} {conf:<8.1f} {'Detection Rate':<20} "
#                     f"{clean_det_rate*100:<12.1f}% {blurry_det_rate*100:<12.1f}% {sp_det_rate*100:<12.1f}% "
#                     f"{blur_det_acc:<12.1f}% {sp_det_acc:<12.1f}%\n"
#                 )
#                 f.write("-"*100 + "\n")

#         print(f"\n✓ Noise robustness comparison saved to: {report_file}")
        
#         # Also print to console
#         print("\n" + "="*100)
#         print("NOISE ROBUSTNESS COMPARISON")
#         print("="*100)
#         with open(report_file, 'r') as f:
#             print(f.read())

#     # ==============================================================================
#     # END OF NEW FUNCTIONALITY
#     # ==============================================================================

#     def run_gunicorn_benchmarks(
#         self,
#         test_pdf_path: str,
#         worker_counts: list = None,
#         num_requests: int = 50,
#         concurrent_requests: int = 10
#     ):
#         """Run uvicorn performance benchmarks"""
#         print("\n" + "="*70)
#         print("UVICORN PERFORMANCE BENCHMARKS")
#         print("="*70)
        
#         if worker_counts is None:
#             worker_counts = [1, 2, 4, 8]
        
#         benchmarker = GunicornBenchmarker(
#             app_module="app.main:app",
#             test_pdf_path=test_pdf_path,
#             output_dir=str(self.output_dir / "uvicorn")
#         )
        
#         results = benchmarker.run_worker_comparison(
#             worker_counts=worker_counts,
#             num_requests=num_requests,
#             concurrent=concurrent_requests
#         )
        
#         output_file = (
#             self.output_dir / "uvicorn" / 
#             f"benchmark_{self.timestamp}.json"
#         )
#         benchmarker.save_results(results, str(output_file))
        
#         # Create text-based comparison
#         self._create_performance_table(results)
        
#         return results
    
#     # ... (rest of the original class methods like _create_accuracy_table, _create_performance_table, generate_report remain the same) ...
#     def _create_accuracy_table(self, results: list):
#         """Create text-based accuracy comparison"""
#         table_file = self.output_dir / f"accuracy_comparison_{self.timestamp}.txt"
        
#         with open(table_file, 'w') as f:
#             f.write("="*80 + "\n")
#             f.write("ACCURACY EVALUATION RESULTS\n")
#             f.write("="*80 + "\n\n")
            
#             # Header
#             f.write(f"{'Model':<12} {'Conf':<8} {'Avg Fields':<12} {'Detection':<12} {'Avg Time':<12} {'Total PDFs':<12}\n")
#             f.write("-"*80 + "\n")
            
#             # Data rows
#             for result in results:
#                 metrics = result['metrics']
#                 det_rate = f"{metrics.avg_detection_rate*100:.1f}%" if metrics.avg_detection_rate else "N/A"
#                 f.write(
#                     f"{result['model']:<12} "
#                     f"{result['confidence']:<8.1f} "
#                     f"{metrics.avg_fields_per_pdf:<12.2f} "
#                     f"{det_rate:<12} "
#                     f"{metrics.avg_processing_time:<12.2f} "
#                     f"{metrics.total_pdfs:<12}\n"
#                 )
            
#             f.write("\n" + "="*80 + "\n")
#             f.write("SUMMARY\n")
#             f.write("="*80 + "\n\n")
            
#             # Find best configurations
#             best_accuracy = max(
#                 results, 
#                 key=lambda x: x['metrics'].avg_detection_rate or 0
#             )
#             fastest = min(results, key=lambda x: x['metrics'].avg_processing_time)
#             most_fields = max(results, key=lambda x: x['metrics'].avg_fields_per_pdf)
            
#             f.write(f"Best accuracy: {best_accuracy['model']} at confidence={best_accuracy['confidence']}\n")
#             if best_accuracy['metrics'].avg_detection_rate:
#                 f.write(f"  Detection rate: {best_accuracy['metrics'].avg_detection_rate*100:.1f}%\n")
#             f.write(f"\nFastest: {fastest['model']} at confidence={fastest['confidence']}\n")
#             f.write(f"  Avg time: {fastest['metrics'].avg_processing_time:.2f}s\n")
#             f.write(f"\nMost fields detected: {most_fields['model']} at confidence={most_fields['confidence']}\n")
#             f.write(f"  Avg fields: {most_fields['metrics'].avg_fields_per_pdf:.2f}\n")
        
#         print(f"\n✓ Accuracy comparison saved to: {table_file}")
        
#         # Also print to console
#         print("\n" + "="*80)
#         print("ACCURACY COMPARISON")
#         print("="*80)
#         with open(table_file, 'r') as f:
#             print(f.read())
    
#     def _create_performance_table(self, results: list):
#         """Create text-based performance comparison"""
#         table_file = self.output_dir / f"performance_comparison_{self.timestamp}.txt"
        
#         with open(table_file, 'w') as f:
#             f.write("="*100 + "\n")
#             f.write("PERFORMANCE EVALUATION RESULTS\n")
#             f.write("="*100 + "\n\n")
            
#             # Header
#             f.write(f"{'Workers':<10} {'RPS':<12} {'Avg Time':<12} {'P95 Time':<12} {'P99 Time':<12} {'CPU %':<10} {'Memory MB':<12}\n")
#             f.write("-"*100 + "\n")
            
#             # Data rows
#             for result in results:
#                 f.write(
#                     f"{result.worker_config.workers:<10} "
#                     f"{result.requests_per_second:<12.2f} "
#                     f"{result.avg_response_time:<12.2f} "
#                     f"{result.p95_response_time:<12.2f} "
#                     f"{result.p99_response_time:<12.2f} "
#                     f"{result.cpu_usage_percent:<10.1f} "
#                     f"{result.memory_usage_mb:<12.1f}\n"
#                 )
            
#             f.write("\n" + "="*100 + "\n")
#             f.write("ANALYSIS\n")
#             f.write("="*100 + "\n\n")
            
#             # Find best configurations
#             best_throughput = max(results, key=lambda x: x.requests_per_second)
#             best_latency = min(results, key=lambda x: x.avg_response_time)
            
#             f.write(f"Best throughput: {best_throughput.worker_config.workers} workers\n")
#             f.write(f"  RPS: {best_throughput.requests_per_second:.2f}\n")
#             f.write(f"  Avg latency: {best_throughput.avg_response_time:.2f}s\n")
#             f.write(f"  P95 latency: {best_throughput.p95_response_time:.2f}s\n")
#             f.write(f"\nBest latency: {best_latency.worker_config.workers} workers\n")
#             f.write(f"  Avg latency: {best_latency.avg_response_time:.2f}s\n")
#             f.write(f"  RPS: {best_latency.requests_per_second:.2f}\n")
            
#             # Calculate scaling efficiency
#             if len(results) > 1:
#                 baseline = results[0]
#                 f.write(f"\nScaling Efficiency (vs 1 worker baseline):\n")
#                 for result in results:
#                     speedup = result.requests_per_second / baseline.requests_per_second
#                     efficiency = (speedup / result.worker_config.workers) * 100
#                     f.write(
#                         f"  {result.worker_config.workers} workers: "
#                         f"{speedup:.2f}x speedup, {efficiency:.1f}% efficient\n"
#                     )
        
#         print(f"\n✓ Performance comparison saved to: {table_file}")
        
#         # Also print to console
#         print("\n" + "="*100)
#         print("PERFORMANCE COMPARISON")
#         print("="*100)
#         with open(table_file, 'r') as f:
#             print(f.read())
    
#     def generate_report(self, accuracy_results: list = None, performance_results: list = None):
#         """Generate comprehensive evaluation report"""
#         report_path = self.output_dir / f"evaluation_report_{self.timestamp}.md"
        
#         with open(report_path, 'w') as f:
#             f.write("# CommonForms API Evaluation Report\n\n")
#             f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
#             # Accuracy section
#             if accuracy_results:
#                 f.write("## Accuracy Evaluation\n\n")
#                 f.write("### Summary\n\n")
#                 f.write("| Model | Confidence | Avg Fields | Detection Rate | Avg Time (s) |\n")
#                 f.write("|-------|-----------|------------|----------------|-------------|\n")
                
#                 for result in accuracy_results:
#                     metrics = result['metrics']
#                     det_rate = f"{metrics.avg_detection_rate*100:.1f}%" if metrics.avg_detection_rate else "N/A"
#                     f.write(
#                         f"| {result['model']} | {result['confidence']} | "
#                         f"{metrics.avg_fields_per_pdf:.2f} | "
#                         f"{det_rate} | "
#                         f"{metrics.avg_processing_time:.2f} |\n"
#                     )
                
#                 f.write("\n### Key Findings\n\n")
                
#                 best_accuracy = max(
#                     accuracy_results, 
#                     key=lambda x: x['metrics'].avg_detection_rate or 0
#                 )
#                 if best_accuracy['metrics'].avg_detection_rate:
#                     f.write(
#                         f"- **Best accuracy**: {best_accuracy['model']} "
#                         f"with confidence={best_accuracy['confidence']} "
#                         f"({best_accuracy['metrics'].avg_detection_rate*100:.1f}% detection rate)\n"
#                     )
                
#                 fastest = min(accuracy_results, key=lambda x: x['metrics'].avg_processing_time)
#                 f.write(
#                     f"- **Fastest processing**: {fastest['model']} "
#                     f"with confidence={fastest['confidence']} "
#                     f"({fastest['metrics'].avg_processing_time:.2f}s)\n"
#                 )
                
#                 most_fields = max(accuracy_results, key=lambda x: x['metrics'].avg_fields_per_pdf)
#                 f.write(
#                     f"- **Most fields detected**: {most_fields['model']} "
#                     f"with confidence={most_fields['confidence']} "
#                     f"({most_fields['metrics'].avg_fields_per_pdf:.2f} avg)\n"
#                 )
            
#             # Performance section
#             if performance_results:
#                 f.write("\n## Gunicorn Performance\n\n")
#                 f.write("### Worker Configuration Results\n\n")
#                 f.write("| Workers | RPS | Avg Time (s) | P95 Time (s) | CPU % | Memory (MB) |\n")
#                 f.write("|---------|-----|--------------|--------------|-------|-------------|\n")
                
#                 for result in performance_results:
#                     f.write(
#                         f"| {result.worker_config.workers} | "
#                         f"{result.requests_per_second:.2f} | "
#                         f"{result.avg_response_time:.2f} | "
#                         f"{result.p95_response_time:.2f} | "
#                         f"{result.cpu_usage_percent:.1f} | "
#                         f"{result.memory_usage_mb:.1f} |\n"
#                     )
                
#                 f.write("\n### Key Findings\n\n")
#                 best_throughput = max(performance_results, key=lambda x: x.requests_per_second)
#                 f.write(
#                     f"- **Best throughput**: {best_throughput.worker_config.workers} workers "
#                     f"({best_throughput.requests_per_second:.2f} req/s)\n"
#                 )
                
#                 best_latency = min(performance_results, key=lambda x: x.avg_response_time)
#                 f.write(
#                     f"- **Best latency**: {best_latency.worker_config.workers} workers "
#                     f"({best_latency.avg_response_time:.2f}s average)\n"
#                 )
                
#                 if len(performance_results) > 1:
#                     baseline = performance_results[0]
#                     best = max(performance_results, key=lambda x: x.requests_per_second)
#                     speedup = best.requests_per_second / baseline.requests_per_second
#                     f.write(
#                         f"- **Speedup**: {speedup:.2f}x with "
#                         f"{best.worker_config.workers} workers vs 1 worker\n"
#                     )
            
#             f.write("\n## Recommendations\n\n")
#             f.write("Based on the evaluation results:\n\n")
            
#             if accuracy_results and performance_results:
#                 # Balanced recommendation
#                 optimal_perf = max(performance_results, key=lambda x: x.requests_per_second)
#                 best_acc = max(accuracy_results, key=lambda x: x['metrics'].avg_detection_rate or 0)
                
#                 f.write("### Production Configuration\n\n")
#                 f.write(f"**For balanced performance:**\n")
#                 f.write(f"- Model: {best_acc['model']}\n")
#                 f.write(f"- Confidence: {best_acc['confidence']}\n")
#                 f.write(f"- Workers: {optimal_perf.worker_config.workers}\n")
#                 f.write(f"- Expected throughput: {optimal_perf.requests_per_second:.2f} req/s\n")
#                 f.write(f"- Expected latency: {optimal_perf.avg_response_time:.2f}s (avg), "
#                        f"{optimal_perf.p95_response_time:.2f}s (P95)\n")
        
#         print(f"\n✓ Report saved to {report_path}")
#         return report_path


# def main():
#     parser = argparse.ArgumentParser(description='Run CommonForms API evaluation (no plots)')
#     parser.add_argument('--api-url', default='http://localhost:8000', 
#                        help='API base URL')
#     parser.add_argument('--username', default='admin', help='API username')
#     parser.add_argument('--password', default='changeme123', help='API password')
    
#     # --- MODIFIED ARGUMENTS ---
#     # Original arguments for single-run accuracy
#     parser.add_argument('--test-pdf-dir', 
#                        help='Directory containing test PDFs for standard accuracy evaluation')
#     # New arguments for noise robustness
#     parser.add_argument('--clean-pdf-dir', 
#                        help='Directory containing clean PDFs for noise robustness evaluation')
#     parser.add_argument('--blurry-pdf-dir', 
#                        help='Directory containing blurry PDFs for noise robustness evaluation')
#     parser.add_argument('--salt-pepper-pdf-dir', 
#                        help='Directory containing salt & pepper PDFs for noise robustness evaluation')

#     parser.add_argument('--test-pdf', required=True, 
#                        help='Single PDF file for performance testing')
#     parser.add_argument('--models', nargs='+', default=['FFDNet-L', 'FFDNet-S'],
#                        help='Models to test')
#     parser.add_argument('--confidences', nargs='+', type=float, 
#                        default=[0.2, 0.3, 0.4],
#                        help='Confidence thresholds to test')
#     parser.add_argument('--workers', nargs='+', type=int, default=[1, 2, 4, 8],
#                        help='Worker counts to test')
#     parser.add_argument('--num-requests', type=int, default=50,
#                        help='Number of requests for performance test')
#     parser.add_argument('--concurrent', type=int, default=10,
#                        help='Concurrent requests')
#     parser.add_argument('--skip-accuracy', action='store_true',
#                        help='Skip standard accuracy evaluation')
#     parser.add_argument('--skip-performance', action='store_true',
#                        help='Skip performance evaluation')
    
#     args = parser.parse_args()
    
#     suite = EvaluationSuite()
    
#     # Get auth token
#     import requests
#     print(f"\nAuthenticating with API at {args.api_url}...")
#     try:
#         response = requests.post(
#             f'{args.api_url}/api/v1/auth/login',
#             json={'username': args.username, 'password': args.password}
#         )
#         if response.status_code != 200:
#             print(f"❌ Authentication failed: {response.status_code}")
#             print(f"Response: {response.text}")
#             sys.exit(1)
#         token = response.json()['access_token']
#         print("✓ Authentication successful")
#     except Exception as e:
#         print(f"❌ Error authenticating: {e}")
#         sys.exit(1)
    
#     accuracy_results = None
#     performance_results = None
    
#     # --- MODIFIED LOGIC IN MAIN ---
#     # Check for noise robustness mode first
#     if args.clean_pdf_dir and args.blurry_pdf_dir and args.salt_pepper_pdf_dir:
#         suite.run_noise_robustness_evaluation(
#             api_url=args.api_url,
#             token=token,
#             clean_pdf_dir=args.clean_pdf_dir,
#             blurry_pdf_dir=args.blurry_pdf_dir,
#             salt_pepper_pdf_dir=args.salt_pepper_pdf_dir,
#             models=args.models,
#             confidences=args.confidences
#         )
#     # Otherwise, run the original standard accuracy evaluation
#     elif not args.skip_accuracy and args.test_pdf_dir:
#         try:
#             accuracy_results = suite.run_accuracy_evaluation(
#                 api_url=args.api_url,
#                 token=token,
#                 test_pdf_dir=args.test_pdf_dir,
#                 models=args.models,
#                 confidences=args.confidences
#             )
#         except Exception as e:
#             print(f"\n❌ Accuracy evaluation failed: {e}")
#             import traceback
#             traceback.print_exc()
    
#     # Run performance evaluation (this remains unchanged)
#     if not args.skip_performance:
#         try:
#             performance_results = suite.run_gunicorn_benchmarks(
#                 test_pdf_path=args.test_pdf,
#                 worker_counts=args.workers,
#                 num_requests=args.num_requests,
#                 concurrent_requests=args.concurrent
#             )
#         except Exception as e:
#             print(f"\n❌ Performance evaluation failed: {e}")
#             import traceback
#             traceback.print_exc()
    
#     # Generate report (this remains unchanged)
#     if accuracy_results or performance_results:
#         try:
#             suite.generate_report(accuracy_results, performance_results)
#         except Exception as e:
#             print(f"\n❌ Report generation failed: {e}")
#             import traceback
#             traceback.print_exc()
    
#     print("\n" + "="*70)
#     print("EVALUATION COMPLETE")
#     print("="*70)
#     print(f"Results saved to: {suite.output_dir}")
#     print(f"\nCheck:")
#     print(f"  - {suite.output_dir}/accuracy/")
#     print(f"  - {suite.output_dir}/gunicorn/")
#     print(f"  - {suite.output_dir}/*.txt (comparison tables)")
#     print(f"  - {suite.output_dir}/*.md (report)")


# if __name__ == "__main__":
#     main()


"""
Complete evaluation suite for CommonForms API (without matplotlib dependency)
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from accuracy_eval import AccuracyEvaluator
from gunicorn_benchmarker import GunicornBenchmarker

class EvaluationSuite:
    """Complete evaluation suite (no plotting)"""
    
    def __init__(self, output_dir: str = "evaluation/complete_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_accuracy_evaluation(
        self,
        api_url: str,
        token: str,
        test_pdf_dir: str,
        models: list = None,
        confidences: list = None,
        device: str = None  # <--- ADDED DEVICE ARGUMENT
    ):
        """Run comprehensive accuracy evaluation"""
        print("\n" + "="*70)
        print(f"ACCURACY EVALUATION ON: {test_pdf_dir}")
        print("="*70)
        
        if models is None:
            models = ["FFDNet-L", "FFDNet-S"]
        if confidences is None:
            confidences = [0.2, 0.3, 0.4]
        
        evaluator = AccuracyEvaluator(
            api_base_url=api_url,
            token=token,
            output_dir=str(self.output_dir / "accuracy")
        )
        
        all_results = []
        
        for model in models:
            for confidence in confidences:
                print(f"\n--- Testing {model} with confidence={confidence} ---")
                
                metrics = evaluator.evaluate_dataset(
                    pdf_dir=test_pdf_dir,
                    model=model,
                    confidence=confidence,
                    device=device # <--- PASSED DEVICE TO EVALUATOR
                )
                
                result = {
                    'model': model,
                    'confidence': confidence,
                    'metrics': metrics
                }
                all_results.append(result)
                
                # Save individual result
                output_file = (
                    self.output_dir / "accuracy" / 
                    f"{model}_{confidence}_{self.timestamp}.json"
                )
                evaluator.save_results(metrics, str(output_file))
        
        return all_results
    
    # ==============================================================================
    # NEW FUNCTIONALITY: Noise Robustness Evaluation
    # ==============================================================================
    def run_noise_robustness_evaluation(
        self,
        api_url: str,
        token: str,
        clean_pdf_dir: str,
        blurry_pdf_dir: str,
        salt_pepper_pdf_dir: str,
        models: list = None,
        confidences: list = None,
        device: str = None # <--- ADDED DEVICE ARGUMENT
    ):
        """Run accuracy evaluation on clean, blurry, and salt&pepper PDFs and compare."""
        print("\n" + "="*70)
        print("NOISE ROBUSTNESS EVALUATION")
        print("="*70)
        
        # 1. Run evaluation on all three PDF sets
        print("\n--- Step 1: Evaluating Clean PDFs ---")
        clean_results = self.run_accuracy_evaluation(
            api_url, token, clean_pdf_dir, models, confidences, device=device # <--- PASSED DEVICE
        )
        
        print("\n--- Step 2: Evaluating Blurry PDFs ---")
        blurry_results = self.run_accuracy_evaluation(
            api_url, token, blurry_pdf_dir, models, confidences, device=device # <--- PASSED DEVICE
        )

        print("\n--- Step 3: Evaluating Salt & Pepper PDFs ---")
        sp_results = self.run_accuracy_evaluation(
            api_url, token, salt_pepper_pdf_dir, models, confidences, device=device # <--- PASSED DEVICE
        )
        
        # 2. Generate the comparison report
        self._create_noise_comparison_report(clean_results, blurry_results, sp_results)
        
        return clean_results, blurry_results, sp_results

    def _create_noise_comparison_report(self, clean_results, blurry_results, sp_results):
        """Create a detailed report comparing clean vs. noisy results."""
        report_file = self.output_dir / f"noise_robustness_report_{self.timestamp}.txt"
        
        # Create lookup maps for easy access to metrics by (model, confidence)
        clean_map = {(r['model'], r['confidence']): r['metrics'] for r in clean_results}
        blurry_map = {(r['model'], r['confidence']): r['metrics'] for r in blurry_results}
        sp_map = {(r['model'], r['confidence']): r['metrics'] for r in sp_results}

        with open(report_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("NOISE ROBUSTNESS EVALUATION REPORT\n")
            f.write("="*100 + "\n\n")
            
            # Header for the comparison table
            f.write(f"{'Model':<12} {'Conf':<8} {'Metric':<20} {'Clean':<12} {'Blur':<12} {'S&P':<12} {'Blur Acc':<12} {'S&P Acc':<12}\n")
            f.write("-"*100 + "\n")
            
            # Iterate through each configuration and compare
            for key, clean_metrics in clean_map.items():
                model, conf = key
                blurry_metrics = blurry_map.get(key)
                sp_metrics = sp_map.get(key)

                if not blurry_metrics or not sp_metrics:
                    continue

                # Compare Average Fields Per PDF
                clean_fields = clean_metrics.avg_fields_per_pdf
                blurry_fields = blurry_metrics.avg_fields_per_pdf
                sp_fields = sp_metrics.avg_fields_per_pdf
                
                blur_field_acc = (blurry_fields / clean_fields * 100) if clean_fields > 0 else 0
                sp_field_acc = (sp_fields / clean_fields * 100) if clean_fields > 0 else 0
                
                f.write(
                    f"{model:<12} {conf:<8.1f} {'Avg Fields':<20} "
                    f"{clean_fields:<12.2f} {blurry_fields:<12.2f} {sp_fields:<12.2f} "
                    f"{blur_field_acc:<12.1f}% {sp_field_acc:<12.1f}%\n"
                )

                # Compare Detection Rate
                clean_det_rate = clean_metrics.avg_detection_rate or 0
                blurry_det_rate = blurry_metrics.avg_detection_rate or 0
                sp_det_rate = sp_metrics.avg_detection_rate or 0

                blur_det_acc = (blurry_det_rate / clean_det_rate * 100) if clean_det_rate > 0 else 0
                sp_det_acc = (sp_det_rate / clean_det_rate * 100) if clean_det_rate > 0 else 0
                
                f.write(
                    f"{model:<12} {conf:<8.1f} {'Detection Rate':<20} "
                    f"{clean_det_rate*100:<12.1f}% {blurry_det_rate*100:<12.1f}% {sp_det_rate*100:<12.1f}% "
                    f"{blur_det_acc:<12.1f}% {sp_det_acc:<12.1f}%\n"
                )
                f.write("-"*100 + "\n")

        print(f"\n✓ Noise robustness comparison saved to: {report_file}")
        
        # Also print to console
        print("\n" + "="*100)
        print("NOISE ROBUSTNESS COMPARISON")
        print("="*100)
        with open(report_file, 'r') as f:
            print(f.read())

    # ==============================================================================
    # END OF NEW FUNCTIONALITY
    # ==============================================================================

    def run_gunicorn_benchmarks(
        self,
        test_pdf_path: str,
        worker_counts: list = None,
        num_requests: int = 50,
        concurrent_requests: int = 10
    ):
        """Run uvicorn performance benchmarks"""
        print("\n" + "="*70)
        print("UVICORN PERFORMANCE BENCHMARKS")
        print("="*70)
        
        if worker_counts is None:
            worker_counts = [1, 2, 4, 8]
        
        benchmarker = GunicornBenchmarker(
            app_module="app.main:app",
            test_pdf_path=test_pdf_path,
            output_dir=str(self.output_dir / "uvicorn")
        )
        
        results = benchmarker.run_worker_comparison(
            worker_counts=worker_counts,
            num_requests=num_requests,
            concurrent=concurrent_requests
        )
        
        output_file = (
            self.output_dir / "uvicorn" / 
            f"benchmark_{self.timestamp}.json"
        )
        benchmarker.save_results(results, str(output_file))
        
        # Create text-based comparison
        self._create_performance_table(results)
        
        return results
    
    # ... (rest of the original class methods like _create_accuracy_table, _create_performance_table, generate_report remain the same) ...
    def _create_accuracy_table(self, results: list):
        """Create text-based accuracy comparison"""
        table_file = self.output_dir / f"accuracy_comparison_{self.timestamp}.txt"
        
        with open(table_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ACCURACY EVALUATION RESULTS\n")
            f.write("="*80 + "\n\n")
            
            # Header
            f.write(f"{'Model':<12} {'Conf':<8} {'Avg Fields':<12} {'Detection':<12} {'Avg Time':<12} {'Total PDFs':<12}\n")
            f.write("-"*80 + "\n")
            
            # Data rows
            for result in results:
                metrics = result['metrics']
                det_rate = f"{metrics.avg_detection_rate*100:.1f}%" if metrics.avg_detection_rate else "N/A"
                f.write(
                    f"{result['model']:<12} "
                    f"{result['confidence']:<8.1f} "
                    f"{metrics.avg_fields_per_pdf:<12.2f} "
                    f"{det_rate:<12} "
                    f"{metrics.avg_processing_time:<12.2f} "
                    f"{metrics.total_pdfs:<12}\n"
                )
            
            f.write("\n" + "="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Find best configurations
            best_accuracy = max(
                results, 
                key=lambda x: x['metrics'].avg_detection_rate or 0
            )
            fastest = min(results, key=lambda x: x['metrics'].avg_processing_time)
            most_fields = max(results, key=lambda x: x['metrics'].avg_fields_per_pdf)
            
            f.write(f"Best accuracy: {best_accuracy['model']} at confidence={best_accuracy['confidence']}\n")
            if best_accuracy['metrics'].avg_detection_rate:
                f.write(f"  Detection rate: {best_accuracy['metrics'].avg_detection_rate*100:.1f}%\n")
            f.write(f"\nFastest: {fastest['model']} at confidence={fastest['confidence']}\n")
            f.write(f"  Avg time: {fastest['metrics'].avg_processing_time:.2f}s\n")
            f.write(f"\nMost fields detected: {most_fields['model']} at confidence={most_fields['confidence']}\n")
            f.write(f"  Avg fields: {most_fields['metrics'].avg_fields_per_pdf:.2f}\n")
        
        print(f"\n✓ Accuracy comparison saved to: {table_file}")
        
        # Also print to console
        print("\n" + "="*80)
        print("ACCURACY COMPARISON")
        print("="*80)
        with open(table_file, 'r') as f:
            print(f.read())
    
    def _create_performance_table(self, results: list):
        """Create text-based performance comparison"""
        table_file = self.output_dir / f"performance_comparison_{self.timestamp}.txt"
        
        with open(table_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("PERFORMANCE EVALUATION RESULTS\n")
            f.write("="*100 + "\n\n")
            
            # Header
            f.write(f"{'Workers':<10} {'RPS':<12} {'Avg Time':<12} {'P95 Time':<12} {'P99 Time':<12} {'CPU %':<10} {'Memory MB':<12}\n")
            f.write("-"*100 + "\n")
            
            # Data rows
            for result in results:
                f.write(
                    f"{result.worker_config.workers:<10} "
                    f"{result.requests_per_second:<12.2f} "
                    f"{result.avg_response_time:<12.2f} "
                    f"{result.p95_response_time:<12.2f} "
                    f"{result.p99_response_time:<12.2f} "
                    f"{result.cpu_usage_percent:<10.1f} "
                    f"{result.memory_usage_mb:<12.1f}\n"
                )
            
            f.write("\n" + "="*100 + "\n")
            f.write("ANALYSIS\n")
            f.write("="*100 + "\n\n")
            
            # Find best configurations
            best_throughput = max(results, key=lambda x: x.requests_per_second)
            best_latency = min(results, key=lambda x: x.avg_response_time)
            
            f.write(f"Best throughput: {best_throughput.worker_config.workers} workers\n")
            f.write(f"  RPS: {best_throughput.requests_per_second:.2f}\n")
            f.write(f"  Avg latency: {best_throughput.avg_response_time:.2f}s\n")
            f.write(f"  P95 latency: {best_throughput.p95_response_time:.2f}s\n")
            f.write(f"\nBest latency: {best_latency.worker_config.workers} workers\n")
            f.write(f"  Avg latency: {best_latency.avg_response_time:.2f}s\n")
            f.write(f"  RPS: {best_latency.requests_per_second:.2f}\n")
            
            # Calculate scaling efficiency
            if len(results) > 1:
                baseline = results[0]
                f.write(f"\nScaling Efficiency (vs 1 worker baseline):\n")
                for result in results:
                    speedup = result.requests_per_second / baseline.requests_per_second
                    efficiency = (speedup / result.worker_config.workers) * 100
                    f.write(
                        f"  {result.worker_config.workers} workers: "
                        f"{speedup:.2f}x speedup, {efficiency:.1f}% efficient\n"
                    )
        
        print(f"\n✓ Performance comparison saved to: {table_file}")
        
        # Also print to console
        print("\n" + "="*100)
        print("PERFORMANCE COMPARISON")
        print("="*100)
        with open(table_file, 'r') as f:
            print(f.read())
    
    def generate_report(self, accuracy_results: list = None, performance_results: list = None):
        """Generate comprehensive evaluation report"""
        report_path = self.output_dir / f"evaluation_report_{self.timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write("# CommonForms API Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Accuracy section
            if accuracy_results:
                f.write("## Accuracy Evaluation\n\n")
                f.write("### Summary\n\n")
                f.write("| Model | Confidence | Avg Fields | Detection Rate | Avg Time (s) |\n")
                f.write("|-------|------------|------------|----------------|-------------|\n")
                
                for result in accuracy_results:
                    metrics = result['metrics']
                    det_rate = f"{metrics.avg_detection_rate*100:.1f}%" if metrics.avg_detection_rate else "N/A"
                    f.write(
                        f"| {result['model']} | {result['confidence']} | "
                        f"{metrics.avg_fields_per_pdf:.2f} | "
                        f"{det_rate} | "
                        f"{metrics.avg_processing_time:.2f} |\n"
                    )
                
                f.write("\n### Key Findings\n\n")
                
                best_accuracy = max(
                    accuracy_results, 
                    key=lambda x: x['metrics'].avg_detection_rate or 0
                )
                if best_accuracy['metrics'].avg_detection_rate:
                    f.write(
                        f"- **Best accuracy**: {best_accuracy['model']} "
                        f"with confidence={best_accuracy['confidence']} "
                        f"({best_accuracy['metrics'].avg_detection_rate*100:.1f}% detection rate)\n"
                    )
                
                fastest = min(accuracy_results, key=lambda x: x['metrics'].avg_processing_time)
                f.write(
                    f"- **Fastest processing**: {fastest['model']} "
                    f"with confidence={fastest['confidence']} "
                    f"({fastest['metrics'].avg_processing_time:.2f}s)\n"
                )
                
                most_fields = max(accuracy_results, key=lambda x: x['metrics'].avg_fields_per_pdf)
                f.write(
                    f"- **Most fields detected**: {most_fields['model']} "
                    f"with confidence={most_fields['confidence']} "
                    f"({most_fields['metrics'].avg_fields_per_pdf:.2f} avg)\n"
                )
            
            # Performance section
            if performance_results:
                f.write("\n## Gunicorn Performance\n\n")
                f.write("### Worker Configuration Results\n\n")
                f.write("| Workers | RPS | Avg Time (s) | P95 Time (s) | CPU % | Memory (MB) |\n")
                f.write("|---------|-----|--------------|--------------|-------|-------------|\n")
                
                for result in performance_results:
                    f.write(
                        f"| {result.worker_config.workers} | "
                        f"{result.requests_per_second:.2f} | "
                        f"{result.avg_response_time:.2f} | "
                        f"{result.p95_response_time:.2f} | "
                        f"{result.cpu_usage_percent:.1f} | "
                        f"{result.memory_usage_mb:.1f} |\n"
                    )
                
                f.write("\n### Key Findings\n\n")
                best_throughput = max(performance_results, key=lambda x: x.requests_per_second)
                f.write(
                    f"- **Best throughput**: {best_throughput.worker_config.workers} workers "
                    f"({best_throughput.requests_per_second:.2f} req/s)\n"
                )
                
                best_latency = min(performance_results, key=lambda x: x.avg_response_time)
                f.write(
                    f"- **Best latency**: {best_latency.worker_config.workers} workers "
                    f"({best_latency.avg_response_time:.2f}s average)\n"
                )
                
                if len(performance_results) > 1:
                    baseline = performance_results[0]
                    best = max(performance_results, key=lambda x: x.requests_per_second)
                    speedup = best.requests_per_second / baseline.requests_per_second
                    f.write(
                        f"- **Speedup**: {speedup:.2f}x with "
                        f"{best.worker_config.workers} workers vs 1 worker\n"
                    )
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on the evaluation results:\n\n")
            
            if accuracy_results and performance_results:
                # Balanced recommendation
                optimal_perf = max(performance_results, key=lambda x: x.requests_per_second)
                best_acc = max(accuracy_results, key=lambda x: x['metrics'].avg_detection_rate or 0)
                
                f.write("### Production Configuration\n\n")
                f.write(f"**For balanced performance:**\n")
                f.write(f"- Model: {best_acc['model']}\n")
                f.write(f"- Confidence: {best_acc['confidence']}\n")
                f.write(f"- Workers: {optimal_perf.worker_config.workers}\n")
                f.write(f"- Expected throughput: {optimal_perf.requests_per_second:.2f} req/s\n")
                f.write(f"- Expected latency: {optimal_perf.avg_response_time:.2f}s (avg), "
                        f"{optimal_perf.p95_response_time:.2f}s (P95)\n")
        
        print(f"\n✓ Report saved to {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Run CommonForms API evaluation (no plots)')
    parser.add_argument('--api-url', default='http://localhost:8000', 
                        help='API base URL')
    parser.add_argument('--username', default='admin', help='API username')
    parser.add_argument('--password', default='changeme123', help='API password')
    
    # --- MODIFIED ARGUMENTS ---
    # Original arguments for single-run accuracy
    parser.add_argument('--test-pdf-dir', 
                        help='Directory containing test PDFs for standard accuracy evaluation')
    # New arguments for noise robustness
    parser.add_argument('--clean-pdf-dir', 
                        help='Directory containing clean PDFs for noise robustness evaluation')
    parser.add_argument('--blurry-pdf-dir', 
                        help='Directory containing blurry PDFs for noise robustness evaluation')
    parser.add_argument('--salt-pepper-pdf-dir', 
                        help='Directory containing salt & pepper PDFs for noise robustness evaluation')

    parser.add_argument('--test-pdf', required=True, 
                        help='Single PDF file for performance testing')
    parser.add_argument('--models', nargs='+', default=['FFDNet-L', 'FFDNet-S'],
                        help='Models to test')
    parser.add_argument('--confidences', nargs='+', type=float, 
                        default=[0.2, 0.3, 0.4],
                        help='Confidence thresholds to test')
    parser.add_argument('--workers', nargs='+', type=int, default=[1, 2, 4, 8],
                        help='Worker counts to test')
    parser.add_argument('--num-requests', type=int, default=50,
                        help='Number of requests for performance test')
    parser.add_argument('--concurrent', type=int, default=10,
                        help='Concurrent requests')
    parser.add_argument('--skip-accuracy', action='store_true',
                        help='Skip standard accuracy evaluation')
    parser.add_argument('--skip-performance', action='store_true',
                        help='Skip performance evaluation')
    
    # <--- ADDED DEVICE ARGUMENT HERE
    parser.add_argument('--device', default='cpu',
                        help='Specify processing device (e.g., cpu, cuda:0) for the API calls. Default is cpu.')
    
    args = parser.parse_args()
    
    suite = EvaluationSuite()
    
    # Get auth token
    import requests
    print(f"\nAuthenticating with API at {args.api_url}...")
    try:
        response = requests.post(
            f'{args.api_url}/api/v1/auth/login',
            json={'username': args.username, 'password': args.password}
        )
        if response.status_code != 200:
            print(f"❌ Authentication failed: {response.status_code}")
            print(f"Response: {response.text}")
            sys.exit(1)
        token = response.json()['access_token']
        print("✓ Authentication successful")
    except Exception as e:
        print(f"❌ Error authenticating: {e}")
        sys.exit(1)
    
    accuracy_results = None
    performance_results = None
    
    # --- MODIFIED LOGIC IN MAIN ---
    # Check for noise robustness mode first
    if args.clean_pdf_dir and args.blurry_pdf_dir and args.salt_pepper_pdf_dir:
        suite.run_noise_robustness_evaluation(
            api_url=args.api_url,
            token=token,
            clean_pdf_dir=args.clean_pdf_dir,
            blurry_pdf_dir=args.blurry_pdf_dir,
            salt_pepper_pdf_dir=args.salt_pepper_pdf_dir,
            models=args.models,
            confidences=args.confidences,
            device=args.device # <--- PASSED DEVICE ARGUMENT
        )
    # Otherwise, run the original standard accuracy evaluation
    elif not args.skip_accuracy and args.test_pdf_dir:
        try:
            accuracy_results = suite.run_accuracy_evaluation(
                api_url=args.api_url,
                token=token,
                test_pdf_dir=args.test_pdf_dir,
                models=args.models,
                confidences=args.confidences,
                device=args.device # <--- PASSED DEVICE ARGUMENT
            )
        except Exception as e:
            print(f"\n❌ Accuracy evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run performance evaluation (this remains unchanged)
    if not args.skip_performance:
        try:
            performance_results = suite.run_gunicorn_benchmarks(
                test_pdf_path=args.test_pdf,
                worker_counts=args.workers,
                num_requests=args.num_requests,
                concurrent_requests=args.concurrent
            )
        except Exception as e:
            print(f"\n❌ Performance evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate report (this remains unchanged)
    if accuracy_results or performance_results:
        try:
            suite.generate_report(accuracy_results, performance_results)
        except Exception as e:
            print(f"\n❌ Report generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {suite.output_dir}")
    print(f"\nCheck:")
    print(f"  - {suite.output_dir}/accuracy/")
    print(f"  - {suite.output_dir}/gunicorn/")
    print(f"  - {suite.output_dir}/*.txt (comparison tables)")
    print(f"  - {suite.output_dir}/*.md (report)")


if __name__ == "__main__":
    main()