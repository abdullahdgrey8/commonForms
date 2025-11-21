from pydantic import BaseModel, Field
from typing import Literal


class PDFProcessRequest(BaseModel):
    """PDF processing request parameters"""
    model: Literal["FFDNet-L", "FFDNet-S"] = Field(
        default="FFDNet-L",
        description="Model to use for field detection"
    )
    keep_existing: bool = Field(
        default=False,
        description="Keep fields that already exist in the PDF"
    )
    use_signature_fields: bool = Field(
        default=False,
        description="Use signature-type fields for detected signatures"
    )
    device: str = Field(
        default="cpu",
        description="Device to run inference on (cpu, cuda:0, etc.)"
    )
    confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold"
    )
    fast: bool = Field(
        default=False,
        description="Use faster ONNX model on CPU"
    )
    multiline: bool = Field(
        default=False,
        description="Allow multiline text boxes"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "FFDNet-L",
                    "keep_existing": False,
                    "use_signature_fields": False,
                    "device": "cpu",
                    "confidence": 0.3,
                    "fast": False,
                    "multiline": False
                }
            ]
        }
    }


class ProcessingStatus(BaseModel):
    """Processing status response"""
    status: str
    message: str
    filename: str | None = None