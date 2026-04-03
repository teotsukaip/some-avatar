#!/usr/bin/env python3
"""
Identify a female pro golfer from a photo by comparing with reference images.

Uses the llava vision model via Ollama API to compare clothing, hairstyle,
and body type (not facial recognition) against reference images.

Usage:
    python identify_golfer.py

Requires:
    - Reference images in the same directory:
      AT.jpg, BT.jpg, CT.jpg, DT.jpg, ET.jpg, FT.jpg, GT.jpg, HT.jpg, IT.png
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
    "IT.png",
]

TARGET_IMAGE = "111111.jpg"


def get_script_dir():
    """Return the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


def load_image_as_base64(filepath):
    """Load an image file and return its base64-encoded string."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(reference_ids):
    """Build the English prompt for the LLM."""
    id_list = ", ".join(
        f"Image {i + 1} = {rid}" for i, rid in enumerate(reference_ids)
    )
    total = len(reference_ids) + 1

    prompt = (
        f"You are given {total} images. "
        f"Images 1 through {len(reference_ids)} are reference photos of different female pro golfers. "
        f"Each reference image has an ID: {id_list}. "
        f"Image {total} is the target photo of an unknown female pro golfer. "
        "Your task is to determine which reference golfer the person in the target photo "
        "most closely resembles. "
        "IMPORTANT: Do NOT use facial features for comparison. "
        "Instead, compare ONLY based on clothing style, outfit colors, hairstyle, "
        "hair color, body type, and overall appearance. "
        "After your analysis, you MUST output your final answer on the last line "
        "in exactly this format:\n"
        "RESULT: <ID>\n"
        "For example: RESULT: AT\n"
        "Replace <ID> with the ID of the most similar reference golfer."
    )
    return prompt


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
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        print(
            "Make sure the Ollama server is running at "
            f"{API_URL} with the {MODEL} model loaded.",
            file=sys.stderr,
        )
        sys.exit(1)


def extract_result(response_text, valid_ids):
    """Extract the golfer ID from the model's response."""
    # Look for "RESULT: XX" pattern in the response
    for line in reversed(response_text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("RESULT:"):
            candidate = line.split(":", 1)[1].strip().upper()
            if candidate in valid_ids:
                return candidate

    # Fallback: search for any valid ID as a whole word in the response
    # Use word-boundary regex to avoid false matches (e.g. "AT" in "THAT")
    # Check from the end of the response (more likely to be the conclusion)
    for line in reversed(response_text.strip().splitlines()):
        for vid in sorted(valid_ids):
            if re.search(r'\b' + re.escape(vid) + r'\b', line.upper()):
                return vid

    return None


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

    # Build reference IDs (filename without extension)
    reference_ids = [os.path.splitext(img)[0] for img in REFERENCE_IMAGES]

    # Load all images as base64
    print("Loading reference images...")
    images_base64 = []
    for img in REFERENCE_IMAGES:
        path = os.path.join(script_dir, img)
        images_base64.append(load_image_as_base64(path))
        print(f"  Loaded: {img} (ID: {os.path.splitext(img)[0]})")

    print(f"Loading target image: {TARGET_IMAGE}")
    images_base64.append(load_image_as_base64(target_path))

    # Build prompt
    prompt = build_prompt(reference_ids)
    print(f"\nSending {len(images_base64)} images to {MODEL} via {API_URL}...")
    print("This may take a while depending on your hardware.\n")

    # Call the API
    result = call_ollama_api(prompt, images_base64)

    response_text = result.get("response", "")
    print("=== Model Response ===")
    print(response_text)
    print("======================\n")

    # Extract the identified golfer ID
    valid_ids = list(reference_ids)
    golfer_id = extract_result(response_text, valid_ids)

    if golfer_id:
        print(f"Identified golfer ID: {golfer_id}")
    else:
        print(
            "Could not determine a clear match from the model's response.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
