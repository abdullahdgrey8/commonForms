# # evaluation/accuracy_evaluator.py
# """
# Evaluate CommonForms field detection accuracy
# """
# import json
# import time
# from pathlib import Path
# from typing import Dict, List, Tuple
# import PyPDF2
# from dataclasses import dataclass, asdict
# import requests
# from datetime import datetime

# @dataclass
# class FieldDetectionResult:
#     """Result of field detection for a single PDF"""
#     filename: str
#     total_fields_detected: int
#     field_types: Dict[str, int]  # Count by type (text, checkbox, signature, etc.)
#     processing_time: float
#     file_size_bytes: int
#     page_count: int
#     fields_per_page: float
#     success: bool
#     error_message: str = None
    
#     # Ground truth comparison (if available)
#     ground_truth_fields: int = None
#     detection_rate: float = None  # detected / ground_truth
#     false_positives: int = None
#     false_negatives: int = None


# @dataclass
# class AccuracyMetrics:
#     """Overall accuracy metrics"""
#     total_pdfs: int
#     successful_pdfs: int
#     failed_pdfs: int
#     total_fields_detected: int
#     avg_fields_per_pdf: float
#     avg_processing_time: float
#     avg_detection_rate: float = None
#     field_type_distribution: Dict[str, int] = None


# class AccuracyEvaluator:
#     """Evaluate field detection accuracy"""
    
#     def __init__(self, api_base_url: str, token: str, output_dir: str = "evaluation/results"):
#         self.api_base_url = api_base_url.rstrip('/')
#         self.token = token
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.results: List[FieldDetectionResult] = []
        
#     def count_pdf_fields(self, pdf_path: str) -> Tuple[int, Dict[str, int], int]:
#         """
#         Count form fields in a PDF
        
#         Returns:
#             (total_fields, field_types_count, page_count)
#         """
#         try:
#             with open(pdf_path, 'rb') as f:
#                 reader = PyPDF2.PdfReader(f)
#                 page_count = len(reader.pages)
                
#                 field_types = {}
#                 total_fields = 0
                
#                 # Get form fields
#                 if reader.get_fields():
#                     for field_name, field_info in reader.get_fields().items():
#                         total_fields += 1
#                         field_type = field_info.get('/FT', 'Unknown')
                        
#                         # Map PDF field types to readable names
#                         type_map = {
#                             '/Tx': 'text',
#                             '/Btn': 'button/checkbox',
#                             '/Ch': 'choice',
#                             '/Sig': 'signature'
#                         }
                        
#                         readable_type = type_map.get(field_type, str(field_type))
#                         field_types[readable_type] = field_types.get(readable_type, 0) + 1
                
#                 return total_fields, field_types, page_count
                
#         except Exception as e:
#             print(f"Error counting fields in {pdf_path}: {e}")
#             return 0, {}, 0
    
#     def evaluate_single_pdf(
#         self, 
#         pdf_path: str,
#         model: str = "FFDNet-L",
#         confidence: float = 0.3,
#         ground_truth_fields: int = None
#     ) -> FieldDetectionResult:
#         """
#         Process a single PDF and evaluate field detection
        
#         Args:
#             pdf_path: Path to input PDF
#             model: Model to use
#             confidence: Detection confidence threshold
#             ground_truth_fields: Expected number of fields (if known)
#         """
#         pdf_path = Path(pdf_path)
#         file_size = pdf_path.stat().st_size
        
#         print(f"\nEvaluating: {pdf_path.name}")
#         print(f"File size: {file_size / 1024:.1f} KB")
        
#         try:
#             # Process PDF through API
#             start_time = time.time()
            
#             with open(pdf_path, 'rb') as f:
#                 files = {'pdf': (pdf_path.name, f, 'application/pdf')}
#                 data = {
#                     'model': model,
#                     'confidence': confidence,
#                     'track_metrics': 'true'
#                 }
#                 headers = {'Authorization': f'Bearer {self.token}'}
                
#                 response = requests.post(
#                     f'{self.api_base_url}/api/v1/pdf/make-fillable',
#                     files=files,
#                     data=data,
#                     headers=headers
#                 )
            
#             processing_time = time.time() - start_time
            
#             if response.status_code != 200:
#                 return FieldDetectionResult(
#                     filename=pdf_path.name,
#                     total_fields_detected=0,
#                     field_types={},
#                     processing_time=processing_time,
#                     file_size_bytes=file_size,
#                     page_count=0,
#                     fields_per_page=0,
#                     success=False,
#                     error_message=f"API error: {response.status_code}"
#                 )
            
#             # Save output PDF
#             output_path = self.output_dir / f"output_{pdf_path.name}"
#             with open(output_path, 'wb') as f:
#                 f.write(response.content)
            
#             # Count detected fields
#             total_fields, field_types, page_count = self.count_pdf_fields(str(output_path))
            
#             fields_per_page = total_fields / page_count if page_count > 0 else 0
            
#             # Calculate accuracy metrics if ground truth available
#             detection_rate = None
#             false_negatives = None
#             if ground_truth_fields is not None:
#                 detection_rate = total_fields / ground_truth_fields if ground_truth_fields > 0 else 0
#                 false_negatives = max(0, ground_truth_fields - total_fields)
            
#             result = FieldDetectionResult(
#                 filename=pdf_path.name,
#                 total_fields_detected=total_fields,
#                 field_types=field_types,
#                 processing_time=processing_time,
#                 file_size_bytes=file_size,
#                 page_count=page_count,
#                 fields_per_page=round(fields_per_page, 2),
#                 success=True,
#                 ground_truth_fields=ground_truth_fields,
#                 detection_rate=round(detection_rate, 3) if detection_rate else None,
#                 false_negatives=false_negatives
#             )
            
#             print(f"✓ Detected {total_fields} fields in {processing_time:.2f}s")
#             print(f"  Field types: {field_types}")
#             if ground_truth_fields:
#                 print(f"  Detection rate: {detection_rate*100:.1f}% ({total_fields}/{ground_truth_fields})")
            
#             return result
            
#         except Exception as e:
#             print(f"✗ Error: {e}")
#             return FieldDetectionResult(
#                 filename=pdf_path.name,
#                 total_fields_detected=0,
#                 field_types={},
#                 processing_time=0,
#                 file_size_bytes=file_size,
#                 page_count=0,
#                 fields_per_page=0,
#                 success=False,
#                 error_message=str(e)
#             )
    
#     def evaluate_dataset(
#         self,
#         pdf_dir: str,
#         model: str = "FFDNet-L",
#         confidence: float = 0.3,
#         ground_truth_file: str = None
#     ) -> AccuracyMetrics:
#         """
#         Evaluate accuracy on a dataset of PDFs
        
#         Args:
#             pdf_dir: Directory containing test PDFs
#             model: Model to use
#             confidence: Detection confidence
#             ground_truth_file: Optional JSON file with ground truth field counts
#         """
#         pdf_dir = Path(pdf_dir)
        
#         # Load ground truth if available
#         ground_truth = {}
#         if ground_truth_file and Path(ground_truth_file).exists():
#             with open(ground_truth_file, 'r') as f:
#                 ground_truth = json.load(f)
        
#         # Process all PDFs
#         pdf_files = list(pdf_dir.glob("*.pdf"))
#         print(f"\n{'='*60}")
#         print(f"Evaluating {len(pdf_files)} PDFs")
#         print(f"Model: {model}, Confidence: {confidence}")
#         print(f"{'='*60}")
        
#         self.results = []
#         for pdf_path in pdf_files:
#             gt_fields = ground_truth.get(pdf_path.name)
#             result = self.evaluate_single_pdf(
#                 str(pdf_path),
#                 model=model,
#                 confidence=confidence,
#                 ground_truth_fields=gt_fields
#             )
#             self.results.append(result)
        
#         # Calculate overall metrics
#         successful = [r for r in self.results if r.success]
        
#         total_fields = sum(r.total_fields_detected for r in successful)
#         avg_fields = total_fields / len(successful) if successful else 0
#         avg_time = sum(r.processing_time for r in successful) / len(successful) if successful else 0
        
#         # Detection rate (only for PDFs with ground truth)
#         detection_rates = [r.detection_rate for r in successful if r.detection_rate is not None]
#         avg_detection_rate = sum(detection_rates) / len(detection_rates) if detection_rates else None
        
#         # Field type distribution
#         field_type_dist = {}
#         for result in successful:
#             for field_type, count in result.field_types.items():
#                 field_type_dist[field_type] = field_type_dist.get(field_type, 0) + count
        
#         metrics = AccuracyMetrics(
#             total_pdfs=len(self.results),
#             successful_pdfs=len(successful),
#             failed_pdfs=len(self.results) - len(successful),
#             total_fields_detected=total_fields,
#             avg_fields_per_pdf=round(avg_fields, 2),
#             avg_processing_time=round(avg_time, 2),
#             avg_detection_rate=round(avg_detection_rate, 3) if avg_detection_rate else None,
#             field_type_distribution=field_type_dist
#         )
        
#         return metrics
    
#     def save_results(self, metrics: AccuracyMetrics, output_file: str = None):
#         """Save evaluation results to JSON"""
#         if output_file is None:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             output_file = self.output_dir / f"accuracy_results_{timestamp}.json"
#         else:
#             output_file = Path(output_file)
        
#         data = {
#             'evaluation_timestamp': datetime.now().isoformat(),
#             'overall_metrics': asdict(metrics),
#             'individual_results': [asdict(r) for r in self.results]
#         }
        
#         with open(output_file, 'w') as f:
#             json.dump(data, f, indent=2)
        
#         print(f"\n{'='*60}")
#         print(f"OVERALL RESULTS")
#         print(f"{'='*60}")
#         print(f"Total PDFs: {metrics.total_pdfs}")
#         print(f"Successful: {metrics.successful_pdfs}")
#         print(f"Failed: {metrics.failed_pdfs}")
#         print(f"Total fields detected: {metrics.total_fields_detected}")
#         print(f"Avg fields per PDF: {metrics.avg_fields_per_pdf}")
#         print(f"Avg processing time: {metrics.avg_processing_time:.2f}s")
#         if metrics.avg_detection_rate:
#             print(f"Avg detection rate: {metrics.avg_detection_rate*100:.1f}%")
#         print(f"\nField type distribution:")
#         for field_type, count in (metrics.field_type_distribution or {}).items():
#             print(f"  {field_type}: {count}")
#         print(f"\nResults saved to: {output_file}")
#         print(f"{'='*60}\n")


# if __name__ == "__main__":
#     # Example usage
#     evaluator = AccuracyEvaluator(
#         api_base_url="http://localhost:8000",
#         token="YOUR_TOKEN_HERE"
#     )
    
#     # Evaluate dataset
#     metrics = evaluator.evaluate_dataset(
#         pdf_dir="test_pdfs",
#         model="FFDNet-L",
#         confidence=0.3,
#         ground_truth_file="test_pdfs/ground_truth.json"
#     )
    
#     evaluator.save_results(metrics)
# evaluation/accuracy_evaluator.py
"""
Evaluate CommonForms field detection accuracy
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import PyPDF2
from dataclasses import dataclass, asdict
import requests
from datetime import datetime

@dataclass
class FieldDetectionResult:
    """Result of field detection for a single PDF"""
    filename: str
    total_fields_detected: int
    field_types: Dict[str, int]  # Count by type (text, checkbox, signature, etc.)
    processing_time: float
    file_size_bytes: int
    page_count: int
    fields_per_page: float
    success: bool
    error_message: str = None
    
    # Ground truth comparison (if available)
    ground_truth_fields: int = None
    detection_rate: float = None  # detected / ground_truth
    false_positives: int = None
    false_negatives: int = None


@dataclass
class AccuracyMetrics:
    """Overall accuracy metrics"""
    total_pdfs: int
    successful_pdfs: int
    failed_pdfs: int
    total_fields_detected: int
    avg_fields_per_pdf: float
    avg_processing_time: float
    avg_detection_rate: float = None
    field_type_distribution: Dict[str, int] = None


class AccuracyEvaluator:
    """Evaluate field detection accuracy"""
    
    def __init__(self, api_base_url: str, token: str, output_dir: str = "evaluation/results"):
        self.api_base_url = api_base_url.rstrip('/')
        self.token = token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[FieldDetectionResult] = []
        
    def count_pdf_fields(self, pdf_path: str) -> Tuple[int, Dict[str, int], int]:
        """
        Count form fields in a PDF
        
        Returns:
            (total_fields, field_types_count, page_count)
        """
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                page_count = len(reader.pages)
                
                field_types = {}
                total_fields = 0
                
                # Get form fields
                if reader.get_fields():
                    for field_name, field_info in reader.get_fields().items():
                        total_fields += 1
                        field_type = field_info.get('/FT', 'Unknown')
                        
                        # Map PDF field types to readable names
                        type_map = {
                            '/Tx': 'text',
                            '/Btn': 'button/checkbox',
                            '/Ch': 'choice',
                            '/Sig': 'signature'
                        }
                        
                        readable_type = type_map.get(field_type, str(field_type))
                        field_types[readable_type] = field_types.get(readable_type, 0) + 1
                
                return total_fields, field_types, page_count
                
        except Exception as e:
            print(f"Error counting fields in {pdf_path}: {e}")
            return 0, {}, 0
    
    def evaluate_single_pdf(
        self, 
        pdf_path: str,
        model: str = "FFDNet-L",
        confidence: float = 0.3,
        ground_truth_fields: int = None,
        device: str = None # <--- ADDED DEVICE ARGUMENT
    ) -> FieldDetectionResult:
        """
        Process a single PDF and evaluate field detection
        
        Args:
            pdf_path: Path to input PDF
            model: Model to use
            confidence: Detection confidence threshold
            ground_truth_fields: Expected number of fields (if known)
            device: Optional device specification (e.g., 'cuda:0', 'cpu')
        """
        pdf_path = Path(pdf_path)
        file_size = pdf_path.stat().st_size
        
        print(f"\nEvaluating: {pdf_path.name}")
        print(f"File size: {file_size / 1024:.1f} KB")
        
        try:
            # Process PDF through API
            start_time = time.time()
            
            with open(pdf_path, 'rb') as f:
                files = {'pdf': (pdf_path.name, f, 'application/pdf')}
                data = {
                    'model': model,
                    'confidence': confidence,
                    'track_metrics': 'true'
                }
                
                # <--- ADDED LOGIC TO INCLUDE DEVICE IN API DATA
                if device:
                    data['device'] = device
                # END OF ADDED LOGIC
                    
                headers = {'Authorization': f'Bearer {self.token}'}
                
                response = requests.post(
                    f'{self.api_base_url}/api/v1/pdf/make-fillable',
                    files=files,
                    data=data,
                    headers=headers
                )
            
            processing_time = time.time() - start_time
            
            if response.status_code != 200:
                return FieldDetectionResult(
                    filename=pdf_path.name,
                    total_fields_detected=0,
                    field_types={},
                    processing_time=processing_time,
                    file_size_bytes=file_size,
                    page_count=0,
                    fields_per_page=0,
                    success=False,
                    error_message=f"API error: {response.status_code}"
                )
            
            # Save output PDF
            output_path = self.output_dir / f"output_{pdf_path.name}"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Count detected fields
            total_fields, field_types, page_count = self.count_pdf_fields(str(output_path))
            
            fields_per_page = total_fields / page_count if page_count > 0 else 0
            
            # Calculate accuracy metrics if ground truth available
            detection_rate = None
            false_negatives = None
            if ground_truth_fields is not None:
                detection_rate = total_fields / ground_truth_fields if ground_truth_fields > 0 else 0
                false_negatives = max(0, ground_truth_fields - total_fields)
            
            result = FieldDetectionResult(
                filename=pdf_path.name,
                total_fields_detected=total_fields,
                field_types=field_types,
                processing_time=processing_time,
                file_size_bytes=file_size,
                page_count=page_count,
                fields_per_page=round(fields_per_page, 2),
                success=True,
                ground_truth_fields=ground_truth_fields,
                detection_rate=round(detection_rate, 3) if detection_rate else None,
                false_negatives=false_negatives
            )
            
            print(f"✓ Detected {total_fields} fields in {processing_time:.2f}s")
            print(f"  Field types: {field_types}")
            if ground_truth_fields:
                print(f"  Detection rate: {detection_rate*100:.1f}% ({total_fields}/{ground_truth_fields})")
            
            return result
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return FieldDetectionResult(
                filename=pdf_path.name,
                total_fields_detected=0,
                field_types={},
                processing_time=0,
                file_size_bytes=file_size,
                page_count=0,
                fields_per_page=0,
                success=False,
                error_message=str(e)
            )
    
    def evaluate_dataset(
        self,
        pdf_dir: str,
        model: str = "FFDNet-L",
        confidence: float = 0.3,
        ground_truth_file: str = None,
        device: str = None # <--- ADDED DEVICE ARGUMENT
    ) -> AccuracyMetrics:
        """
        Evaluate accuracy on a dataset of PDFs
        
        Args:
            pdf_dir: Directory containing test PDFs
            model: Model to use
            confidence: Detection confidence
            ground_truth_file: Optional JSON file with ground truth field counts
            device: Optional device specification (e.g., 'cuda:0', 'cpu')
        """
        pdf_dir = Path(pdf_dir)
        
        # Load ground truth if available
        ground_truth = {}
        if ground_truth_file and Path(ground_truth_file).exists():
            with open(ground_truth_file, 'r') as f:
                ground_truth = json.load(f)
        
        # Process all PDFs
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"\n{'='*60}")
        print(f"Evaluating {len(pdf_files)} PDFs")
        print(f"Model: {model}, Confidence: {confidence}")
        if device:
            print(f"Device: {device}") # <--- ADDED PRINT FOR NEW DEVICE ARGUMENT
        print(f"{'='*60}")
        
        self.results = []
        for pdf_path in pdf_files:
            gt_fields = ground_truth.get(pdf_path.name)
            result = self.evaluate_single_pdf(
                str(pdf_path),
                model=model,
                confidence=confidence,
                ground_truth_fields=gt_fields,
                device=device # <--- PASSED NEW DEVICE ARGUMENT
            )
            self.results.append(result)
        
        # Calculate overall metrics
        successful = [r for r in self.results if r.success]
        
        total_fields = sum(r.total_fields_detected for r in successful)
        avg_fields = total_fields / len(successful) if successful else 0
        avg_time = sum(r.processing_time for r in successful) / len(successful) if successful else 0
        
        # Detection rate (only for PDFs with ground truth)
        detection_rates = [r.detection_rate for r in successful if r.detection_rate is not None]
        avg_detection_rate = sum(detection_rates) / len(detection_rates) if detection_rates else None
        
        # Field type distribution
        field_type_dist = {}
        for result in successful:
            for field_type, count in result.field_types.items():
                field_type_dist[field_type] = field_type_dist.get(field_type, 0) + count
        
        metrics = AccuracyMetrics(
            total_pdfs=len(self.results),
            successful_pdfs=len(successful),
            failed_pdfs=len(self.results) - len(successful),
            total_fields_detected=total_fields,
            avg_fields_per_pdf=round(avg_fields, 2),
            avg_processing_time=round(avg_time, 2),
            avg_detection_rate=round(avg_detection_rate, 3) if avg_detection_rate else None,
            field_type_distribution=field_type_dist
        )
        
        return metrics
    
    def save_results(self, metrics: AccuracyMetrics, output_file: str = None):
        """Save evaluation results to JSON"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"accuracy_results_{timestamp}.json"
        else:
            output_file = Path(output_file)
        
        data = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'overall_metrics': asdict(metrics),
            'individual_results': [asdict(r) for r in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"OVERALL RESULTS")
        print(f"{'='*60}")
        print(f"Total PDFs: {metrics.total_pdfs}")
        print(f"Successful: {metrics.successful_pdfs}")
        print(f"Failed: {metrics.failed_pdfs}")
        print(f"Total fields detected: {metrics.total_fields_detected}")
        print(f"Avg fields per PDF: {metrics.avg_fields_per_pdf}")
        print(f"Avg processing time: {metrics.avg_processing_time:.2f}s")
        if metrics.avg_detection_rate:
            print(f"Avg detection rate: {metrics.avg_detection_rate*100:.1f}%")
        print(f"\nField type distribution:")
        for field_type, count in (metrics.field_type_distribution or {}).items():
            print(f"  {field_type}: {count}")
        print(f"\nResults saved to: {output_file}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    evaluator = AccuracyEvaluator(
        api_base_url="http://localhost:8000",
        token="YOUR_TOKEN_HERE"
    )
    
    # Evaluate dataset
    metrics = evaluator.evaluate_dataset(
        pdf_dir="test_pdfs",
        model="FFDNet-L",
        confidence=0.3,
        ground_truth_file="test_pdfs/ground_truth.json"
    )
    
    evaluator.save_results(metrics)