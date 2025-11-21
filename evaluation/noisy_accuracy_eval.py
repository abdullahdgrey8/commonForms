# evaluation/noisy_accuracy_eval.py
"""
Noise Robustness Evaluation for CommonForms API
Compares field detection on Clean vs Blurry vs Salt&Pepper PDFs
"""

import os
import sys
import requests
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

API_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "changeme123"

# Directories
CLEAN_DIR = Path("test_pdfs")
BLUR_DIR = Path("blur_pdfs")
SP_DIR = Path("salt_pepper_pdfs")

# Fixed processing parameters
MODEL = "FFDNet-L"
CONFIDENCE = 0.3
DEVICE = "cuda:0"  # Will auto-fallback to CPU if not available
KEEP_EXISTING = False
USE_SIGNATURE = False
FAST = False
MULTILINE = False
TRACK_METRICS = True


class NoiseRobustnessEvaluator:
    def __init__(self, api_url: str, output_dir: str = "evaluation/noise_results"):
        self.api_url = api_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token = self._login()

    def _login(self) -> str:
        print("Logging in...")
        resp = requests.post(
            f"{self.api_url}/api/v1/auth/login",
            json={"username": USERNAME, "password": PASSWORD}
        )
        if resp.status_code != 200:
            raise Exception(f"Login failed: {resp.status_code} {resp.text}")
        token = resp.json()["access_token"]
        print("Login successful")
        return token

    def _get_headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def count_fields_in_pdf(self, pdf_path: Path) -> int:
        """
        Uses your own API to process PDF and count detected fields
        Returns number of fields added
        """
        url = f"{self.api_url}/api/v1/pdf/make-fillable"
        files = {"pdf": (pdf_path.name, open(pdf_path, "rb"), "application/pdf")}
        data = {
            "model": MODEL,
            "confidence": CONFIDENCE,
            "device": DEVICE,
            "keep_existing": str(KEEP_EXISTING).lower(),
            "use_signature_fields": str(USE_SIGNATURE).lower(),
            "fast": str(FAST).lower(),
            "multiline": str(MULTILINE).lower(),
            "track_metrics": str(TRACK_METRICS).lower(),
        }

        print(f"  → Processing {pdf_path.name}...", end="")
        start = time.time()

        resp = requests.post(url, files=files, data=data, headers=self._get_headers(), stream=True)

        if resp.status_code != 200:
            print(f" Failed ({resp.status_code})")
            return 0

        # Save output temporarily to analyze
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
            for chunk in resp.iter_content(chunk_size=8192):
                tmp_out.write(chunk)
            output_path = tmp_out.name

        # Count AcroForm fields using PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(output_path)
            fields = reader.get_fields()
            count = len(fields) if fields else 0
        except Exception as e:
            print(f" Error reading fields: {e}")
            count = 0
        finally:
            os.unlink(output_path)

        duration = time.time() - start
        print(f" {count} fields ({duration:.2f}s)")
        return count

    def evaluate_dataset(self, pdf_dir: Path, label: str) -> Dict:
        print(f"\nEvaluating {label} PDFs from: {pdf_dir}")
        if not pdf_dir.exists():
            print(f"   Directory not found!")
            return {"label": label, "files": [], "total_fields": 0, "avg_fields": 0, "count": 0}

        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print("   No PDFs found.")
            return {"label": label, "files": [], "total_fields": 0, "avg_fields": 0, "count": 0}

        results = []
        total_fields = 0

        for pdf_path in sorted(pdf_files):
            fields = self.count_fields_in_pdf(pdf_path)
            results.append({"filename": pdf_path.name, "fields_detected": fields})
            total_fields += fields

        avg = total_fields / len(pdf_files) if pdf_files else 0

        summary = {
            "label": label,
            "directory": str(pdf_dir),
            "count": len(pdf_files),
            "total_fields": total_fields,
            "avg_fields": round(avg, 2),
            "files": results
        }
        return summary

    def run_full_evaluation(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("="*80)
        print("COMMONFORMS NOISE ROBUSTNESS EVALUATION")
        print("="*80)
        print(f"Model: {MODEL} | Confidence: {CONFIDENCE} | Device: {DEVICE}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Run evaluations
        clean_result = self.evaluate_dataset(CLEAN_DIR, "CLEAN")
        blur_result = self.evaluate_dataset(BLUR_DIR, "BLURRY")
        sp_result = self.evaluate_dataset(SP_DIR, "SALT & PEPPER")

        all_results = [clean_result, blur_result, sp_result]

        # Calculate degradation
        clean_avg = clean_result["avg_fields"]
        blur_drop = ((clean_avg - blur_result["avg_fields"]) / clean_avg * 100) if clean_avg > 0 else 0
        sp_drop = ((clean_avg - sp_result["avg_fields"]) / clean_avg * 100) if clean_avg > 0 else 0

        # Save detailed results
        detail_path = self.output_dir / f"noise_robustness_detailed_{timestamp}.json"
        with open(detail_path, "w") as f:
            json.dump({
                "config": {
                    "model": MODEL,
                    "confidence": CONFIDENCE,
                    "device": DEVICE,
                    "timestamp": datetime.now().isoformat()
                },
                "results": all_results
            }, f, indent=2)
        print(f"\nDetailed results → {detail_path}")

        # Generate beautiful report
        report_path = self.output_dir / f"NOISE_ROBUSTNESS_REPORT_{timestamp}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# CommonForms Noise Robustness Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** `{MODEL}` | **Confidence:** `{CONFIDENCE}` | **Device:** `{DEVICE}`\n\n")
            f.write("## Summary Table\n\n")
            f.write("| Dataset          | PDFs | Avg Fields | vs Clean     | Degradation     |\n")
            f.write("|------------------|------|------------|--------------|-----------------|\n")
            f.write(f"| **Clean**        | {clean_result['count']:<4} | **{clean_avg:.2f}** | baseline     | -               |\n")
            f.write(f"| Blurry           | {blur_result['count']:<4} | {blur_result['avg_fields']:.2f}     | -{blur_drop:.1f}%      | {blur_drop:.1f}% drop   |\n")
            f.write(f"| Salt & Pepper    | {sp_result['count']:<4} | {sp_result['avg_fields']:.2f}     | -{sp_drop:.1f}%       | {sp_drop:.1f}% drop   |\n")

            f.write("\n## Interpretation\n\n")
            if blur_drop < 15 and sp_drop < 25:
                f.write("**Excellent robustness!** Model handles noise very well.\n")
            elif blur_drop < 30 and sp_drop < 50:
                f.write("**Good robustness.** Acceptable drop under heavy noise.\n")
            else:
                f.write("**Warning: Model is sensitive to noise.** Consider denoising preprocessing.\n")

            f.write(f"\n## Recommendation\n")
            if blur_drop > 20 or sp_drop > 40:
                f.write("- Consider adding image preprocessing (denoising + sharpening)\n")
                f.write("- Or fine-tune model on noisy data\n")
            else:
                f.write("- Model is production-ready even on scanned/noisy documents!\n")

        print(f"\nReport saved → {report_path}")

        # Print console summary
        print("\n" + "="*80)
        print("NOISE ROBUSTNESS SUMMARY")
        print("="*80)
        print(f"{'Dataset':<18} {'PDFs':<6} {'Avg Fields':<12} {'Drop vs Clean'}")
        print("-"*80)
        print(f"{'CLEAN':<18} {clean_result['count']:<6} {clean_avg:<12.2f} -")
        print(f"{'BLURRY':<18} {blur_result['count']:<6} {blur_result['avg_fields']:<12.2f} ↓ {blur_drop:.1f}%")
        print(f"{'SALT&PEPPER':<18} {sp_result['count']:<6} {sp_result['avg_fields']:<12.2f} ↓ {sp_drop:.1f}%")
        print("="*80)

        if blur_drop < 20 and sp_drop < 35:
            print("Robust model – works great on real-world scanned PDFs!")
        else:
            print("Consider adding denoising step before processing")

        print(f"\nAll results saved in: {self.output_dir}")
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CommonForms robustness to blur and salt&pepper noise")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--clean-dir", default="test_pdfs", help="Clean PDFs directory")
    parser.add_argument("--blur-dir", default="blur_pdfs", help="Blurred PDFs directory")
    parser.add_argument("--sp-dir", default="salt_pepper_pdfs", help="Salt & pepper PDFs directory")
    args = parser.parse_args()

    global CLEAN_DIR, BLUR_DIR, SP_DIR
    CLEAN_DIR = Path(args.clean_dir)
    BLUR_DIR = Path(args.blur_dir)
    SP_DIR = Path(args.sp_dir)

    evaluator = NoiseRobustnessEvaluator(api_url=args.api_url)
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()