#!/usr/bin/env python3
"""
Identify a female pro golfer from a photo by comparing with reference images.

Uses the llava vision model via Ollama API to compare clothing, hairstyle,
and body type (not facial recognition) against reference images.

Each reference image is compared individually with the target image
(pair-wise comparison) to get a similarity score, then the best match
is selected.

Usage:
    python identify_golfer.py

Requires:
    - Reference images in the same directory:
      AT.jpg, BT.jpg, CT.jpg, DT.jpg, ET.jpg, FT.jpg, GT.jpg, HT.jpg
    - Target image: 111111.jpg
    - Ollama server running at http://localhost:11435 with llava:latest model
"""

import base64
import json
import os
import re
import sys
import urllib.request
import urllib.error

API_URL = "http://localhost:11435/api/generate"
MODEL = "llava:latest"

REFERENCE_IMAGES = [
    "AT.jpg",
    "BT.jpg",
    "CT.jpg",
    "DT.jpg",
    "ET.jpg",
    "FT.jpg",
    "GT.jpg",
    "HT.jpg",
]

TARGET_IMAGE = "111111.jpg"


def get_script_dir():
    """Return the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


def load_image_as_base64(filepath):
    """Load an image file and return its base64-encoded string."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_comparison_prompt():
    """Build the English prompt for pair-wise comparison."""
    return (
        "You are given 2 images of female pro golfers. "
        "Image 1 is a reference photo and Image 2 is a target photo. "
        "Compare the two people based ONLY on: "
        "clothing style, outfit colors, hairstyle, hair color, "
        "body type, and overall appearance. "
        "Do NOT use facial features for comparison. "
        "Rate the similarity on a scale from 0 to 10, where "
        "0 means completely different and 10 means extremely similar. "
        "You MUST output your score on the last line in exactly this format:\n"
        "SCORE: <number>\n"
        "For example: SCORE: 7\n"
        "Replace <number> with your similarity score (0-10)."
    )


def call_ollama_api(prompt, images_base64):
    """Call the Ollama API with the prompt and images."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": images_base64,
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        print(f"HTTP Error {e.code}: {e.reason}", file=sys.stderr)
        print(f"Response: {error_body}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        print(
            "Make sure the Ollama server is running at "
            f"{API_URL} with the {MODEL} model loaded.",
            file=sys.stderr,
        )
        return None


def extract_score(response_text):
    """Extract the similarity score from the model's response."""
    # Look for "SCORE: X" pattern in the response (check from the end)
    for line in reversed(response_text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            score_str = line.split(":", 1)[1].strip()
            match = re.search(r'(\d+(?:\.\d+)?)', score_str)
            if match:
                score = float(match.group(1))
                return min(score, 10.0)

    # Fallback: search for any number after "score" keyword
    for line in reversed(response_text.strip().splitlines()):
        match = re.search(r'(?:score|rating|similarity)\s*[:=]?\s*(\d+(?:\.\d+)?)',
                          line, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return min(score, 10.0)

    # Last fallback: find any number out of 10 (e.g. "7/10")
    for line in reversed(response_text.strip().splitlines()):
        match = re.search(r'\b(\d+(?:\.\d+)?)\s*/\s*10\b', line)
        if match:
            return min(float(match.group(1)), 10.0)

    return -1.0  # Indicate failure


def main():
    script_dir = get_script_dir()

    # Verify all required files exist
    missing = []
    for img in REFERENCE_IMAGES:
        path = os.path.join(script_dir, img)
        if not os.path.isfile(path):
            missing.append(img)

    target_path = os.path.join(script_dir, TARGET_IMAGE)
    if not os.path.isfile(target_path):
        missing.append(TARGET_IMAGE)

    if missing:
        print("Error: The following required image files are missing:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    # Load target image
    print(f"Loading target image: {TARGET_IMAGE}")
    target_base64 = load_image_as_base64(target_path)

    # Build the comparison prompt (same for all pairs)
    prompt = build_comparison_prompt()

    # Compare each reference image with the target
    results = []  # list of (golfer_id, score, response_text)

    for img in REFERENCE_IMAGES:
        golfer_id = os.path.splitext(img)[0]
        path = os.path.join(script_dir, img)
        ref_base64 = load_image_as_base64(path)

        print(f"\nComparing {golfer_id} with target image...")

        # Send reference image + target image as a pair
        api_result = call_ollama_api(prompt, [ref_base64, target_base64])

        if api_result is None:
            print(f"  ERROR: API call failed for {golfer_id}", file=sys.stderr)
            results.append((golfer_id, -1.0, ""))
            continue

        response_text = api_result.get("response", "")
        score = extract_score(response_text)

        print(f"  Response: {response_text.strip()[:200]}")
        print(f"  Score: {score}")

        results.append((golfer_id, score, response_text))

    # Find the best match
    print("\n" + "=" * 50)
    print("Results Summary:")
    print("=" * 50)

    valid_results = [(gid, s, r) for gid, s, r in results if s >= 0]

    if valid_results:
        best_id, best_score, _ = max(valid_results, key=lambda x: x[1])
    else:
        best_id = None

    for golfer_id, score, _ in sorted(results, key=lambda x: x[1], reverse=True):
        score_str = f"{score:.1f}/10" if score >= 0 else "ERROR"
        marker = " <-- BEST MATCH" if golfer_id == best_id else ""
        print(f"  {golfer_id}: {score_str}{marker}")

    if best_id is not None:
        print(f"\nIdentified golfer ID: {best_id} (score: {best_score:.1f}/10)")
    else:
        print(
            "\nCould not determine a match from any comparison.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
