"""Few-shot subclassification with a small instruction-tuned LLM.

The model's output is constrained by deterministic postprocessing, notably the
creative-purpose restriction and conservative handling of unverified identity.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

CLASSES = {"deepfake_real_entity", "deepfake_creative", "general_synthetic", "not_ai_generated"}
SYSTEM_PROMPT = """You classify AI-image compliance context. Do not identify or guess anyone's identity.
 Use only evidence supplied in the inputs. Never infer that an entity is real or that
 content is deceptive solely from photorealism or a caption. If evidence is insufficient,
 use general_synthetic for detected AI content. deepfake_creative is permitted ONLY when
 declared_purpose is exactly 'satire' or 'art', AND evidence supports depiction of a real
 person/entity. deepfake_real_entity requires explicit verified real-entity context.
 Return ONLY JSON: {"subclass":"...", "rationale":"..."}.

Examples:
Input: {"caption":{"scene_summary":"A stylized forest"},"detection":{"label":"ai"},"context":{}}
Output: {"subclass":"general_synthetic","rationale":"Synthetic landscape without verified real-entity context."}
Input: {"caption":{"scene_summary":"A mountain photograph"},"detection":{"label":"real"},"context":{}}
Output: {"subclass":"not_ai_generated","rationale":"Detector labels the image real."}
Input: {"caption":{"scene_summary":"A staged image of a public event"},"detection":{"label":"ai"},"context":{"verified_real_entity":true,"declared_purpose":"satire"}}
Output: {"subclass":"deepfake_creative","rationale":"Declared satire with verified real-entity context."}
Input: {"caption":{"scene_summary":"An altered image depicting a known real event"},"detection":{"label":"ai"},"context":{"verified_real_entity":true,"declared_purpose":"news"}}
Output: {"subclass":"deepfake_real_entity","rationale":"Verified real-world entity or event context and no qualifying creative declaration."}
"""


def _parse(raw: str) -> dict[str, str]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I).strip()
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("subclass") in CLASSES and isinstance(obj.get("rationale"), str):
            return {"subclass": obj["subclass"], "rationale": obj["rationale"]}
    raise ValueError("Invalid classification JSON")


class Classifier:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", generator: Callable | None = None):
        self.model_id = model_id
        self.generator = generator
        self._model = None
        self._tokenizer = None

    def _generate(self, payload: dict) -> str:
        if self.generator is not None:
            return str(self.generator(payload))
    
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            ).eval()
    
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
    
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
    
        inputs = inputs.to(self._model.device)
    
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False
            )
    
        generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    
        return self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

    def classify(self, caption: dict, detection: dict, context: dict | None = None) -> dict[str, str]:
        context = context or {}
        label = str(detection.get("label", "")).lower()
        if label == "real":
            return {"subclass": "not_ai_generated", "rationale": "Detector classified the image as real; not independently verified."}
        if label not in {"ai", "ai_generated", "ai_edited"}:
            raise ValueError("detection.label must be 'real' or an AI label")
        payload = {"caption": caption, "detection": detection, "context": context}
        try:
            result = _parse(self._generate(payload))
        except (ValueError, TypeError):
            return {"subclass": "general_synthetic", "rationale": "Classification output invalid; conservative fallback."}
        chosen = result["subclass"]
        purpose = context.get("declared_purpose")
        verified = context.get("verified_real_entity") is True
        if chosen == "not_ai_generated":
            return {"subclass": "general_synthetic", "rationale": "AI detector label takes precedence over inconsistent subclass output."}
        if chosen == "deepfake_creative" and (purpose not in {"satire", "art"} or not verified):
            return {"subclass": "general_synthetic", "rationale": "Creative deepfake requires satire/art declaration and verified real-entity context."}
        if chosen == "deepfake_real_entity" and not verified:
            return {"subclass": "general_synthetic", "rationale": "Real-entity context is unverified; conservative fallback."}
        return result


def classify(caption: dict, detection: dict, context: dict | None = None,
             classifier: Classifier | None = None) -> dict[str, str]:
    return (classifier or Classifier()).classify(caption, detection, context)
