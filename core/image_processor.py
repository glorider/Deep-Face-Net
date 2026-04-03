import sys
import cv2
from pathlib import Path


def process_image(source_face, face_analyser, args):
    """Process a single image.

    Args:
        mode (via args.mode or args.enhance):
            'swap'         – face swap only (default)
            'enhance'      – GFPGAN on original faces, no swap
            'swap_enhance' – face swap then GFPGAN enhancement
    """
    from core.engine.face_swapper import swap_face

    print(f"Processing image: {args.target}")

    # Load target image
    target_img = cv2.imread(args.target)
    if target_img is None:
        print(f"Error: Failed to load target image: {args.target}")
        sys.exit(1)

    h, w = target_img.shape[:2]

    # Detect faces
    print("Detecting faces...")
    faces = face_analyser.get(target_img)
    print(f"Found {len(faces)} face(s)")
    if not faces:
        print("Warning: No faces detected in target image")

    result = target_img.copy()

    # Step 1: Swap
    if faces:
        for face in faces:
            result = swap_face(source_face, face, result)

    # Save output
    if args.output:
        output_path = args.output
    else:
        target_path = Path(args.target)
        output_path = f"{target_path.stem}_swapped{target_path.suffix}"

    cv2.imwrite(output_path, result)
    print(f"✓ Output saved: {output_path}")
