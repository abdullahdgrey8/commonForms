# scripts/analyze_metrics.py
"""
Utility script to analyze stored metrics

Usage:
    python scripts/analyze_metrics.py
    python scripts/analyze_metrics.py --export-csv metrics.csv
    python scripts/analyze_metrics.py --plot
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.metrics import ProcessingMetrics


class MetricsAnalyzer:
    """Analyze stored processing metrics"""
    
    def __init__(self, metrics_file: str = "data/metrics/processing_metrics.jsonl"):
        self.metrics_file = Path(metrics_file)
        self.metrics: List[ProcessingMetrics] = []
        self.load_metrics()
    
    def load_metrics(self):
        """Load metrics from file"""
        if not self.metrics_file.exists():
            print(f"❌ Metrics file not found: {self.metrics_file}")
            return
        
        with open(self.metrics_file, 'r') as f:
            for line in f:
                try:
                    metric = ProcessingMetrics.model_validate_json(line)
                    self.metrics.append(metric)
                except Exception as e:
                    print(f"Warning: Could not parse line: {e}")
        
        print(f"✅ Loaded {len(self.metrics)} metrics records")
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.metrics:
            return {}
        
        successful = [m for m in self.metrics if m.success]
        failed = [m for m in self.metrics if not m.success]
        
        times = [m.total_processing_time for m in successful]
        
        # GPU metrics
        gpu_used = [m for m in successful if m.gpu_metrics and m.gpu_metrics.gpu_available]
        gpu_memory = [
            m.gpu_metrics.peak_memory_mb 
            for m in gpu_used 
            if m.gpu_metrics.peak_memory_mb
        ]
        
        # DPI metrics
        dpi_tracked = [m for m in successful if m.dpi_metrics]
        input_dpis = [m.dpi_metrics.input_dpi for m in dpi_tracked]
        output_dpis = [m.dpi_metrics.output_dpi for m in dpi_tracked]
        
        # Models used
        models = {}
        for m in successful:
            models[m.model_used] = models.get(m.model_used, 0) + 1
        
        return {
            "total_requests": len(self.metrics),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": f"{(len(successful) / len(self.metrics) * 100):.1f}%",
            "processing_time": {
                "avg": f"{sum(times) / len(times):.2f}s" if times else "N/A",
                "min": f"{min(times):.2f}s" if times else "N/A",
                "max": f"{max(times):.2f}s" if times else "N/A",
            },
            "gpu_usage": {
                "requests_with_gpu": len(gpu_used),
                "avg_peak_memory_mb": f"{sum(gpu_memory) / len(gpu_memory):.2f}" if gpu_memory else "N/A",
                "max_peak_memory_mb": f"{max(gpu_memory):.2f}" if gpu_memory else "N/A",
            },
            "dpi_metrics": {
                "tracked_requests": len(dpi_tracked),
                "avg_input_dpi": f"{sum(input_dpis) / len(input_dpis):.1f}" if input_dpis else "N/A",
                "avg_output_dpi": f"{sum(output_dpis) / len(output_dpis):.1f}" if output_dpis else "N/A",
            },
            "models_used": models,
        }
    
    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()
        
        if not summary:
            print("No metrics to analyze")
            return
        
        print("\n" + "="*60)
        print("📊 METRICS SUMMARY")
        print("="*60)
        
        print(f"\n📈 Overall Statistics:")
        print(f"  Total Requests:    {summary['total_requests']}")
        print(f"  Successful:        {summary['successful']}")
        print(f"  Failed:            {summary['failed']}")
        print(f"  Success Rate:      {summary['success_rate']}")
        
        print(f"\n⏱️  Processing Time:")
        print(f"  Average:           {summary['processing_time']['avg']}")
        print(f"  Minimum:           {summary['processing_time']['min']}")
        print(f"  Maximum:           {summary['processing_time']['max']}")
        
        print(f"\n🎮 GPU Usage:")
        print(f"  Requests with GPU: {summary['gpu_usage']['requests_with_gpu']}")
        print(f"  Avg Peak Memory:   {summary['gpu_usage']['avg_peak_memory_mb']} MB")
        print(f"  Max Peak Memory:   {summary['gpu_usage']['max_peak_memory_mb']} MB")
        
        print(f"\n🖼️  DPI Metrics:")
        print(f"  Tracked Requests:  {summary['dpi_metrics']['tracked_requests']}")
        print(f"  Avg Input DPI:     {summary['dpi_metrics']['avg_input_dpi']}")
        print(f"  Avg Output DPI:    {summary['dpi_metrics']['avg_output_dpi']}")
        
        print(f"\n🤖 Models Used:")
        for model, count in summary['models_used'].items():
            print(f"  {model:15} {count} requests")
        
        print("\n" + "="*60 + "\n")
    
    def export_csv(self, output_file: str):
        """Export metrics to CSV"""
        import csv
        
        if not self.metrics:
            print("No metrics to export")
            return
        
        with open(output_file, 'w', newline='') as f:
            fieldnames = [
                'request_id', 'timestamp', 'filename', 'file_size_bytes',
                'model_used', 'device', 'total_processing_time',
                'success', 'input_dpi', 'output_dpi',
                'gpu_available', 'gpu_peak_memory_mb', 'gpu_memory_increase_mb'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for m in self.metrics:
                row = {
                    'request_id': m.request_id,
                    'timestamp': m.timestamp.isoformat(),
                    'filename': m.filename,
                    'file_size_bytes': m.file_size_bytes,
                    'model_used': m.model_used,
                    'device': m.device,
                    'total_processing_time': m.total_processing_time,
                    'success': m.success,
                    'input_dpi': m.dpi_metrics.input_dpi if m.dpi_metrics else None,
                    'output_dpi': m.dpi_metrics.output_dpi if m.dpi_metrics else None,
                    'gpu_available': m.gpu_metrics.gpu_available if m.gpu_metrics else False,
                    'gpu_peak_memory_mb': m.gpu_metrics.peak_memory_mb if m.gpu_metrics else None,
                    'gpu_memory_increase_mb': m.gpu_metrics.memory_increase_mb if m.gpu_metrics else None,
                }
                writer.writerow(row)
        
        print(f"✅ Exported {len(self.metrics)} records to {output_file}")
    
    def plot_metrics(self):
        """Create visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("❌ matplotlib not installed. Install with: pip install matplotlib")
            return
        
        if not self.metrics:
            print("No metrics to plot")
            return
        
        successful = [m for m in self.metrics if m.success]
        times = [m.total_processing_time for m in successful]
        timestamps = [m.timestamp for m in successful]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('CommonForms API Metrics Analysis', fontsize=16)
        
        # 1. Processing time over time
        axes[0, 0].plot(timestamps, times, 'b-', alpha=0.6)
        axes[0, 0].set_title('Processing Time Over Time')
        axes[0, 0].set_xlabel('Timestamp')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Processing time histogram
        axes[0, 1].hist(times, bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Processing Time Distribution')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. GPU memory usage
        gpu_metrics = [m for m in successful if m.gpu_metrics and m.gpu_metrics.gpu_available]
        if gpu_metrics:
            gpu_memory = [m.gpu_metrics.peak_memory_mb for m in gpu_metrics]
            axes[1, 0].bar(range(len(gpu_memory)), gpu_memory, color='purple', alpha=0.7)
            axes[1, 0].set_title('GPU Peak Memory Usage')
            axes[1, 0].set_xlabel('Request #')
            axes[1, 0].set_ylabel('Memory (MB)')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No GPU metrics available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. DPI comparison
        dpi_metrics = [m for m in successful if m.dpi_metrics]
        if dpi_metrics:
            input_dpis = [m.dpi_metrics.input_dpi for m in dpi_metrics]
            output_dpis = [m.dpi_metrics.output_dpi for m in dpi_metrics]
            
            x = np.arange(len(input_dpis))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, input_dpis, width, label='Input DPI', alpha=0.7)
            axes[1, 1].bar(x + width/2, output_dpis, width, label='Output DPI', alpha=0.7)
            axes[1, 1].set_title('DPI Comparison')
            axes[1, 1].set_xlabel('Request #')
            axes[1, 1].set_ylabel('DPI')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No DPI metrics available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # Save plot
        output_file = 'data/metrics/metrics_plot.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {output_file}")
        
        # Show plot
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze CommonForms API metrics')
    parser.add_argument('--export-csv', type=str, help='Export metrics to CSV file')
    parser.add_argument('--plot', action='store_true', help='Create visualization plots')
    parser.add_argument('--metrics-file', type=str, 
                       default='data/metrics/processing_metrics.jsonl',
                       help='Path to metrics file')
    
    args = parser.parse_args()
    
    analyzer = MetricsAnalyzer(args.metrics_file)
    
    # Always print summary
    analyzer.print_summary()
    
    # Export CSV if requested
    if args.export_csv:
        analyzer.export_csv(args.export_csv)
    
    # Create plots if requested
    if args.plot:
        analyzer.plot_metrics()


if __name__ == '__main__':
    main()