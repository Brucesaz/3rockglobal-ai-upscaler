
# AI Video Upscaler with Real-ESRGAN (3rockglobal)

This Streamlit app allows you to:
- Upscale or downscale videos using Real-ESRGAN
- Choose from multiple models (photo/anime)
- Automatically match output to original framerate and color format
- Pad to target resolution while preserving aspect ratio
- Preview comparison (PIP or side-by-side)
- Detect OS and offer correct launcher

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Run:

```bash
streamlit run app.py
```

Place your model weights in the `weights/` folder (e.g., RealESRGAN_x4plus.pth)

## Platform Notes

- Windows users: Double-click the `.bat` launcher
- macOS users: Use the `.command` launcher (set permission with `chmod +x`)
