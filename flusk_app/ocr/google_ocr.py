import requests

class GoogleVisionOCR:
    def __init__(self, api_key: str):
        self.endpoint = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    def detect_text(self, base64_image: str) -> str:
        # Strip data URL header if present
        if base64_image.strip().startswith("data:"):
            base64_image = base64_image.split(",", 1)[1]

        body = {
            "requests": [
                {
                    "image": {"content": base64_image},
                    "features": [
                        {
                            "type": "DOCUMENT_TEXT_DETECTION",
                            "model": "builtin/latest"  # prefer latest OCR
                        }
                    ],
                    "imageContext": {
                        "languageHints": ["en"]  # bias to English (printed + handwritten)
                    }
                }
            ]
        }

        resp = requests.post(self.endpoint, json=body)
        resp.raise_for_status()
        data = resp.json()

        try:
            return data["responses"][0]["fullTextAnnotation"]["text"]
        except (KeyError, IndexError, TypeError):
            return ""
