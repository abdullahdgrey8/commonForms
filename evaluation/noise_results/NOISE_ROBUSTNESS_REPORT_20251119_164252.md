# CommonForms Noise Robustness Report

**Generated:** 2025-11-19 16:44:19
**Model:** `FFDNet-L` | **Confidence:** `0.3` | **Device:** `cuda:0`

## Summary Table

| Dataset          | PDFs | Avg Fields | vs Clean     | Degradation     |
|------------------|------|------------|--------------|-----------------|
| **Clean**        | 8    | **79.88** | baseline     | -               |
| Blurry           | 8    | 51.25     | -35.8%      | 35.8% drop   |
| Salt & Pepper    | 8    | 0.00     | -100.0%       | 100.0% drop   |

## Interpretation

**Warning: Model is sensitive to noise.** Consider denoising preprocessing.

## Recommendation
- Consider adding image preprocessing (denoising + sharpening)
- Or fine-tune model on noisy data
