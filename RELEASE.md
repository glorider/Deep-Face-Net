# Deep Face Net v2.0.0-rc

Release Candidate for v2.0.0

## What's New

### Model Manager Tab
A brand new **Models** tab in the application for managing AI models directly from the GUI.

- View all required and optional models with download status
- **Download models directly** from the app with live progress bars
- Status indicators: `[OK]` (downloaded, green) / `[!!]` (missing, orange)
- `REQUIRED` / `OPTIONAL` badges for each model
- Shows file size, download path, and current status
- **Startup warning**: Tab title shows `[!] Models` if required models are missing
- **Refresh Status** button to re-check all models
- Re-download and Retry (on error) support
- Models tracked:
  - `inswapper_128.onnx` (Required) - Face swap model (FP32, ~528 MB)
  - `inswapper_128.fp16.onnx` (Optional) - Face swap model (FP16, ~264 MB)
  - `GFPGANv1.4.pth` (Optional) - Face enhancement model (~332 MB)
  - `buffalo_l` (Required) - Face detection & analysis pack (~311 MB)

### Performance Improvements
- **Reworked threading architecture** in the Qt GUI for significantly improved FPS
- Face detection and face swapping now both run in the background `ThreadPoolExecutor`
- Main thread only handles frame capture and display, eliminating the previous bottleneck
- New `detect_and_swap()` function combines detection + swap into a single threaded operation
- FPS improved from 7-8 FPS to 17 FPS

### Application Icon
- New custom app icon: green wireframe polygonal face on transparent background
- Displayed in the window title bar and system taskbar

### Cross-Platform UI Fixes
- Fixed layout issues reported on Windows (Python 3.13, GPU 3050)
- Merged from `fix/ui-cross-platform` branch (PR #3, reported by @glorider)

## Files Changed

| File | Changes |
|---|---|
| `app/deepfake_app.py` | Model Manager tab, ModelDownloadThread, app icon, startup model check |
| `app/video_thread.py` | Reworked threading: full pipeline offloaded to executor |
| `core/engine/face_swapper.py` | Added `detect_and_swap()` function |
| `core/config.py` | Added `INSIGHTFACE_DIR`, `BUFFALO_L_DIR` path constants |
| `download_models.py` | Added `buffalo_l` model, `check_model_status()`, `get_model_path()`, `format_size()` helpers |
| `assets/icon.png` | New application icon |

## Breaking Changes

None. This release is fully backward compatible with v1.0.1.

## Known Issues

- **Chin artifacts**: Visible seam around chin/jawline when mouth masking is disabled. This is a known limitation of the `inswapper_128.onnx` model's paste-back mechanism, not a code bug.

## Requirements

No new dependencies. All existing requirements from v1.0.1 remain the same.

## How to Test

1. Pull the `v2.0.0-rc` branch
2. Run the app: `python -m app.deepfake_app`
3. Check the **Models** tab - verify status indicators and download functionality
4. Start camera - verify improved FPS
5. Check window icon appears in title bar
