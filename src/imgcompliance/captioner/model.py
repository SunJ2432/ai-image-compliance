"""Image description with Qwen3.5-4B. Lazy-loads model on first use.

No person identification: the prompt explicitly prohibits naming or guessing identities.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

FIELDS = ("scene_summary", "depicts_people", "appears_photorealistic", "possible_real_world_reference")
PROMPT = """Describe the image based only on visible evidence. NEVER identify, name, recognize,
 or guess the identity of any person, including public figures. Do not infer personal attributes.
 Do not claim to verify a real-world location or event from appearance alone.
 Return ONLY a JSON object with exactly these fields:
 {"scene_summary": "short factual visual description", "depicts_people": false,
 "appears_photorealistic": true, "possible_real_world_reference": "visible possible place/event reference or null"}.
 The two boolean fields MUST be JSON booleans. Use null if no grounded reference exists.
"""


def parse_caption_json(raw: str) -> dict[str, Any]:
    """Extract and validate the first JSON object, including fenced responses."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or any(key not in value for key in FIELDS):
            continue
        if not isinstance(value["scene_summary"], str) or not value["scene_summary"].strip():
            continue
        if type(value["depicts_people"]) is not bool or type(value["appears_photorealistic"]) is not bool:
            continue
        reference = value["possible_real_world_reference"]
        if reference is not None and not isinstance(reference, str):
            continue
        return {key: value[key] for key in FIELDS}
    raise ValueError("No valid caption JSON found")


class Captioner:
    def __init__(self, model_id: str = "Qwen/Qwen3.5-4B", generator: Callable | None = None,
                 max_new_tokens: int = 180):
        self.model_id = model_id
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _generate(self, image: Any, prompt: str) -> str:
        if self.generator is not None:
            return str(self.generator(image, prompt))
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
        if self._model is None:
            if not torch.cuda.is_available():
                raise RuntimeError("Qwen3.5-4B requires a GPU for this Colab prototype")
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            ).eval()
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                pil_image = opened.convert("RGB")
        else:
            pil_image = image.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil_image}, {"type": "text", "text": prompt}
        ]}]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()

    def describe(self, image: Any) -> dict[str, Any]:
        """Retry invalid JSON once, then return a plain-text fallback with null unknowns."""
        raw = ""
        for attempt in range(2):
            raw = self._generate(image, PROMPT + ("\nIMPORTANT: Correct the previous invalid JSON. JSON only." if attempt else ""))
            try:
                parsed = parse_caption_json(raw)
                return {**parsed, "format": "json", "model_version": self.model_id}
            except ValueError:
                pass
        # Do not hallucinate boolean values when structured parsing fails.
        return {
            "scene_summary": raw.strip(),
            "depicts_people": None,
            "appears_photorealistic": None,
            "possible_real_world_reference": None,
            "format": "plain_text_fallback",
            "model_version": self.model_id,
        }


def describe(image: Any, captioner: Captioner | None = None) -> dict[str, Any]:
    return (captioner or Captioner()).describe(image)
