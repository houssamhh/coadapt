"""
This script is used to feed a scene description to an LLM and parses its decision on:
  (1) which robots (CAVs) participate in the fusion process
  (2) which data fusion strategy to use (early / intermediate / late)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional

# ── make pipeline/ and opencood importable ──────────────────────────────────
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from coadapt.scene_abstraction_module import build_scene_description


# ════════════════════════════════════════════════════════════════════════════
# Bandwidth generation
# ════════════════════════════════════════════════════════════════════════════

# Base bandwidth tiers (Mbps).
#
# High   (50 Mbps): comfortably supports early fusion for a few CAVs
# Medium (20 Mbps): supports compressed intermediate or selective early
# Low    (5 Mbps):  only late fusion or heavily compressed intermediate
_BASE_BW = {"high": 50.0, "medium": 20.0, "low": 5.0}


def generate_bandwidth(scenario_name: str, frame_id: int) -> float:
    """Generate a deterministic, continuous bandwidth value (Mbps).

    The value is fully determined by (scenario_name, frame_id), so it is
    reproducible across experiments without any external random state.

    Components
    ----------
    1. **Base tier** — selected from the hour field in the scenario name:
       - Hour  6–11:  high  (50 Mbps)
       - Hour 12–17:  medium (20 Mbps)
       - Hour 18–23 or 0–5:  low (5 Mbps)

    2. **Sinusoidal variation** — ±15 % of base, period = 50 frames:
       ``0.15 × base × sin(2π × frame_id / 50)``

    Returns
    -------
    float
        Available bandwidth in Mbps (clamped to ≥ 0.1).
    """
    # --- 1. Extract hour from scenario name (format: YYYY_MM_DD_HH_MM_SS) ---
    parts = scenario_name.split("_")
    try:
        hour = int(parts[3])          # e.g. "2021_08_20_20_39_00" → 20
    except (IndexError, ValueError):
        hour = 12                     # fallback → medium

    if 6 <= hour <= 11:
        base = _BASE_BW["high"]
    elif 12 <= hour <= 17:
        base = _BASE_BW["medium"]
    else:
        base = _BASE_BW["low"]

    # --- 2. Sinusoidal component ---
    sinusoidal = 0.15 * base * math.sin(2 * math.pi * frame_id / 50)

    # --- 3. Deterministic noise via MD5 ---
    digest = hashlib.md5(f"{scenario_name}_{frame_id}".encode()).hexdigest()
    hash_int = int(digest[:8], 16)
    noise = (hash_int / 0xFFFFFFFF) * 2 - 1      # ∈ [-1, 1]
    noise_component = 0.05 * base * noise

    return max(0.1, base + sinusoidal + noise_component)


SYSTEM_PROMPT = """\
You are a cooperative-perception orchestrator for a fleet of autonomous vehicles (CAVs).
You receive a scene description and must jointly select:
  (1) which CAVs participate in the fusion process
  (2) which data fusion strategy to use

Goal: maximise perception accuracy while minimising bandwidth usage.

---
HARD CONSTRAINTS
---
  C1. The ego vehicle is always included.
  C2. Any CAV beyond the communication range cannot transmit — exclude it.
  C3. At least one non-ego in-range CAV must be included.

---
INPUTS PROVIDED
---
  - CAV positions (x, y), distance to ego, and compass direction from ego
  - Per-CAV obstacle descriptions derived from their LiDAR point clouds
  - Available bandwidth between each pair of robots (in Mbps)
  - Previous decision and its AP metrics (if available)

---
REASONING GUIDANCE
---
  SPATIAL COVERAGE
  A CAV adds value if it observes a region or obstacles the ego cannot see well.
  Use the obstacle descriptions to assess whether a CAV's view is complementary
  or redundant relative to the ego.  Two CAVs in the same direction with similar
  obstacles provide little additional value.
  Favour subsets that cover diverse directions and distinct obstacle zones.

  BANDWIDTH COST vs AVAILABLE BANDWIDTH
  Fusion strategies differ in cost and accuracy:
    "early"        - raw LiDAR point clouds (~1.8 MB/frame/CAV).  Highest accuracy, highest bandwidth.
    "intermediate" - feature maps (~0.5–1.1 MB/frame/CAV compressed, ~18 MB uncompressed).  Good accuracy, moderate bandwidth.
    "late"         - bounding boxes only (~1 KB/frame/CAV).  Lowest accuracy, lowest bandwidth.
  The available bandwidth is given in Mbps. Compare it against the per-CAV cost
  (multiply by number of non-ego CAVs and the frame rate of 10 Hz) to judge
  whether a strategy is feasible.
  When bandwidth is tight, prefer late or intermediate fusion and fewer CAVs.
  When bandwidth is ample, early fusion with a diverse subset is justified.

  CONTINUITY
  If the scene geometry, obstacle distribution, and network state have not changed much since
  the previous cycle, prefer to keep the same selection unless there is a
  clear reason to change.  Use the previous AP to judge whether the last
  decision was effective.

---
OUTPUT FORMAT
---
Write 3-5 sentences of reasoning, then output the JSON on the very last line.

Rules for the JSON line:
  - It must be valid JSON, nothing else on that line.
  - Do NOT wrap it in markdown code fences (no ```).
  - Do NOT add any text after the closing brace.
  - "fusion_method" must be exactly one of: "early", "intermediate", "late".
  - "selected_cavs" must include the ego vehicle ID.

Example of a correct final line:
{"selected_cavs": ["ego_id", "cav_X"], "fusion_method": "intermediate", "reason": "Diverse coverage under medium congestion."}
"""


_VALID_FUSION = {"early", "intermediate", "late"}


def _try_json(s: str) -> Optional[dict]:
    """Try to parse *s* as JSON; return None on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def parse_response(text: str) -> Dict:
    """Extract the decision JSON from a raw LLM response.

    Tries multiple strategies in order:
      1. Strip markdown code fences, then scan lines from the bottom up.
      2. **Output** tag.
      3. Any JSON object containing 'selected_cavs' (single-line).
      4. Any JSON object containing 'selected_cavs' (multi-line / DOTALL).
    """
    # Pre-process: strip <think>...</think> blocks (Qwen3 / GPT-OSS reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Pre-process: strip markdown code fences (```json ... ``` or ``` ... ```)
    clean = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")

    parsed = None

    # Strategy 1: scan lines from bottom, accept first parseable JSON object
    for line in reversed(clean.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            parsed = _try_json(line)
            if parsed is not None:
                break

    # Strategy 2: **Output** tag (some instruction-tuned models)
    if parsed is None:
        m = re.search(r"\*\*Output\*\*\s*(\{.*?\})", clean, re.DOTALL)
        if m:
            parsed = _try_json(m.group(1))

    # Strategy 3: single-line object containing selected_cavs
    if parsed is None:
        m = re.search(r'\{[^{}\n]*"selected_cavs"[^{}\n]*\}', clean)
        if m:
            parsed = _try_json(m.group(0))

    # Strategy 4: multi-line object containing selected_cavs (DOTALL)
    if parsed is None:
        for m in re.finditer(r'\{[^{}]*"selected_cavs"[^{}]*\}', clean, re.DOTALL):
            parsed = _try_json(m.group(0))
            if parsed is not None:
                break

    # Strategy 5: outermost balanced braces in the whole text
    if parsed is None:
        start = clean.find("{")
        if start != -1:
            depth, end = 0, -1
            for i, ch in enumerate(clean[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                parsed = _try_json(clean[start : end + 1])

    if parsed is None or "selected_cavs" not in parsed:
        return {"selected_cavs": [], "fusion_method": "unknown", "reason": "parse_error"}

    # Normalise fusion_method
    fm = str(parsed.get("fusion_method", "")).strip().lower()
    if fm not in _VALID_FUSION:
        for candidate in _VALID_FUSION:
            if candidate in text.lower():
                fm = candidate
                break
        else:
            fm = "unknown"
    parsed["fusion_method"] = fm
    return parsed


# ════════════════════════════════════════════════════════════════════════════
# LLM backends
# ════════════════════════════════════════════════════════════════════════════

class _AnthropicBackend:
    """Thin wrapper around the Anthropic Messages API.

    Activated automatically when model_id starts with 'claude-'.
    Requires:  pip install anthropic
               ANTHROPIC_API_KEY environment variable
    """

    def __init__(self, model_id: str, max_new_tokens: int = 8192):
        import anthropic
        self._client   = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
        self._model_id = model_id
        self._max_tokens = max_new_tokens
        print(f"[selector] Using Anthropic cloud model: {model_id}")

    def generate(self, user_prompt: str) -> str:
        import anthropic
        response = self._client.messages.create(
            model=self._model_id,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


class _LLMBackend:
    """Thin wrapper around a HuggingFace causal-LM."""

    def __init__(self, model_id: str, hf_token: Optional[str] = None,
                 max_new_tokens: int = 8192, quantization: Optional[int] = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        if hf_token:
            from huggingface_hub import login
            login(hf_token)

        print(f"[selector] Loading model: {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._is_gptoss = "gpt-oss" in model_id.lower()

        if quantization in (4, 8):
            # Use bitsandbytes for 4-bit or 8-bit quantization.
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(quantization == 4),
                load_in_8bit=(quantization == 8),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                torch_dtype=torch.float16 if quantization == 8 else torch.bfloat16,
                device_map="cuda",
            )
            print(f"[selector] Loaded with {quantization}-bit quantization (bitsandbytes).")
        elif quantization == 16:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            print("[selector] Loaded in float16.")
        elif quantization == 32:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="auto",
            )
            print("[selector] Loaded in float32.")
        else:
            # Default: bfloat16 for GPT-OSS, auto for others
            dtype = torch.bfloat16 if self._is_gptoss else "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
            )
            print(f"[selector] Loaded in {'bfloat16' if self._is_gptoss else 'auto'} dtype.")

        self.max_new_tokens = max_new_tokens
        print("[selector] Model loaded.")

    def generate(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if self._is_gptoss:
            chat_kwargs["enable_thinking"] = False
        prompt_text = self.tokenizer.apply_chat_template(messages, **chat_kwargs)
        inputs  = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if self._is_gptoss and "assistantfinal" in text:
            text = text.split("assistantfinal", 1)[-1].strip()
        return text


class RobotAndStrategySelector:
    """Loads an LLM once and exposes a .select() method per frame.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID (e.g. 'google/gemma-3-27b-it').
    hf_token : str or None
        HuggingFace token for gated models.
    max_new_tokens : int
        Maximum tokens to generate per call.
    quantization : int or None
        Quantization level: 4, 8, 16, or 32. None = auto dtype.
        Ignored for GPT-OSS models that use MXFP4 quantization.
    """

    def __init__(self, model_id: str, hf_token: Optional[str] = None,
                 max_new_tokens: int = 8192, quantization: Optional[int] = None):
        if model_id.startswith("claude-"):
            self._llm = _AnthropicBackend(model_id, max_new_tokens)
        else:
            self._llm = _LLMBackend(model_id, hf_token, max_new_tokens,
                                    quantization=quantization)

    def select(
        self,
        scenario_data: dict,
        bandwidth_mbps: float = 50.0,
        prev_result: Optional[dict] = None,
        prev_ap: Optional[dict] = None,
    ) -> Dict:
        """Run the LLM on a single frame and return its decision.

        Returns
        -------
        dict with keys:
            selected_cavs  : list[str]
            fusion_method  : 'early' | 'intermediate' | 'late' | 'unknown'
            reason         : str
        """
        description  = build_scene_description(scenario_data, bandwidth_mbps,
                                               prev_result, prev_ap)
        raw_response = self._llm.generate(description)
        result = parse_response(raw_response)

        # Retry once if the first response failed to parse
        if result["fusion_method"] == "unknown":
            retry_prompt = (
                description
                + "\n\nYour previous response could not be parsed. "
                "Output ONLY the JSON on a single line with no other text:\n"
                '{"selected_cavs": [...], "fusion_method": "early"|"intermediate"|"late", "reason": "..."}'
            )
            raw_response = self._llm.generate(retry_prompt)
            result = parse_response(raw_response)

        # Enforce hard constraints when parse succeeded
        if result.get("fusion_method") != "unknown":
            cavs = result.get("selected_cavs") or []

            # C2: drop any CAV beyond communication range
            cr = scenario_data.get("com_range", float("inf"))
            cav_info = scenario_data.get("cavs", {})
            cavs = [c for c in cavs
                    if c in cav_info and cav_info[c]["dist_to_ego"] <= cr]

            # C1: ego is always included
            ego = scenario_data.get("ego_cav")
            if ego and ego not in cavs:
                cavs = [ego] + cavs

            # C3: at least one non-ego in-range CAV must be selected
            non_ego_selected = [c for c in cavs if c != ego]
            if not non_ego_selected:
                in_range_non_ego = [
                    (c, info["dist_to_ego"]) for c, info in cav_info.items()
                    if c != ego and info.get("dist_to_ego", float("inf")) <= cr
                ]
                if in_range_non_ego:
                    nearest = min(in_range_non_ego, key=lambda x: x[1])[0]
                    cavs.append(nearest)
                    print(f"[selector] C3: forced nearest non-ego CAV {nearest} into selection")

            result["selected_cavs"] = cavs

        return result, raw_response
