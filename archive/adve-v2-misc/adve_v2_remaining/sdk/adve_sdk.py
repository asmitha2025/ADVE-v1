import urllib.request
import urllib.parse
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class EmbedResult:
    status: str
    video_path: str
    index_name: str
    total_frames: int
    mean_cos_sim: float
    min_cos_sim: float
    encoder_savings_pct: float
    effective_fps: float
    processing_time_sec: float
    license_info: Dict[str, Any]

class ADVEClient:
    """
    Official ADVE Enterprise Python Client SDK v3.1
    
    Usage:
        from adve_sdk import ADVEClient
        
        client = ADVEClient(
            base_url="http://localhost:8000",
            license_key="EVAL-TATA-2026"
        )
        
        result = client.embed_video("/data/your_video.mp4")
        print(f"Savings: {result.encoder_savings_pct}%")
        print(f"Accuracy: {result.mean_cos_sim}")
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", license_key: str = "EVAL-DEMO-2026"):
        self.base_url = base_url.rstrip("/")
        self.license_key = license_key

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "X-License-Key": self.license_key
        }

    def health_check(self) -> dict:
        url = f"{self.base_url}/health"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def validate_license(self) -> dict:
        url = f"{self.base_url}/api/v1/license/validate"
        req = urllib.request.Request(url, headers=self._headers(), method="POST")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def embed_video(self, video_path: str, index_name: str = "main_index", max_frames: int = 500) -> EmbedResult:
        url = f"{self.base_url}/api/v1/embed"
        payload = json.dumps({
            "video_path": video_path,
            "index_name": index_name,
            "max_frames": max_frames
        }).encode("utf-8")
        
        req = urllib.request.Request(url, data=payload, headers=self._headers(), method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return EmbedResult(
                status=data.get("status", "unknown"),
                video_path=data.get("video_path", video_path),
                index_name=data.get("index_name", index_name),
                total_frames=data.get("total_frames", 0),
                mean_cos_sim=data.get("mean_cos_sim", 0.0),
                min_cos_sim=data.get("min_cos_sim", 0.0),
                encoder_savings_pct=data.get("encoder_savings_pct", 0.0),
                effective_fps=data.get("effective_fps", 0.0),
                processing_time_sec=data.get("processing_time_sec", 0.0),
                license_info=data.get("license", {})
            )

if __name__ == "__main__":
    client = ADVEClient()
    try:
        health = client.health_check()
        print("ADVE Server Health Check:", health)
    except Exception as e:
        print("ADVE Client SDK Initialization (Server not running locally):", e)
