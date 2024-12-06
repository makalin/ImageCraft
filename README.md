# ImageCraft: Interactive Image Filters

ImageCraft is an interactive application designed for applying basic filters to images. The tool leverages OpenCV for image processing and Gradio for creating an intuitive web-based user interface. Whether you're looking to create artistic edits or perform functional image transformations, ImageCraft simplifies the process with a few clicks.

## Key Features
- **Real-time image filtering**: Upload your image and apply filters instantly.
- **Supported filters**:
  - Grayscale: Converts the image to black and white.
  - Sepia: Adds a warm, vintage tone to your image.
  - X-Ray: Simulates an X-ray effect by inverting colors and enhancing contrasts for a surreal appearance.  
  - Blur: Softens the image by reducing sharp edges and blending colors for a smoother look.  
  - Sharpen: Enhances the edges and details in the image, making it crisper and more defined.  
  - Emboss: Creates a 3D-like texture by highlighting edges and applying a raised effect.  
  - Edge Detection: Identifies and outlines the edges in the image, often used for artistic or analytical purposes.  
  - Mosaic Effect: Breaks the image into blocky patterns, giving it a pixelated, artistic appearance.  
  - Rainbow: Overlays a rainbow spectrum effect, adding vibrant and colorful hues to your image.  
  - Night Vision: Mimics the greenish hue of night vision devices, enhancing low-light details.  
  - Wave Effect: Distorts the image with wavy patterns, creating a surreal and dynamic appearance.  
  - Timestamp: Adds a date and time stamp to the image, commonly used for documentation or retro aesthetics.  
  - Glitch Effect: Introduces digital glitches such as color shifts and distortions for a modern, edgy vibe.  
  - Oil Paint: Transforms the image into a painting-like texture with smooth, blended brushstrokes.  
  - Cartoon: Simplifies details and adds bold outlines, giving the image a hand-drawn, cartoonish appearance.  
  - Nostalgia: Applies a faded, vintage filter with soft tones and slight grain, evoking memories of the past.  
  - Brightness Increase: Enhances the brightness of the image, making it lighter and more vibrant.

- **User-friendly interface**: Gradio provides a simple interface for interacting with the application.

## Getting Started
Follow the steps below to set up and run ImageCraft locally.

### Prerequisites
Make sure you have the following installed on your system:
- Python 3.8 or higher
- `pip` package manager

### Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/imagecraft.git
   cd imagecraft
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open the application in your browser. Gradio will provide a local URL for the interface.

## Usage
1. **Upload Image**: Use the upload button to select an image from your device.
2. **Choose a Filter**: Select one of the available filters (e.g., Grayscale or Sepia).
3. **Preview and Download**: View the transformed image in real time and download it if desired.

## Project Structure
```
imagecraft/
├── app.py                # Main application script
├── README.md             # Project documentation
├── requirements.txt      # List of dependencies
└── example_images/       # Sample images for testing (optional)
```

## Dependencies
The project relies on the following Python libraries:
- [Gradio](https://gradio.app/): For creating the web-based user interface.
- [OpenCV](https://opencv.org/): For image processing.
- [NumPy](https://numpy.org/): For numerical computations.

To install all dependencies, run:
```bash
pip install gradio opencv-python-headless numpy
```

## Example
Below is an example of how the application looks in action:

*(Include screenshots of the Gradio interface and before/after filter examples.)*

## Contributing
Contributions are welcome! If you have ideas for new features or improvements, please:
1. Fork the repository.
2. Create a new branch for your feature.
3. Submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements
Special thanks to:
- [OpenCV](https://opencv.org/) for its powerful image processing capabilities.
- [Gradio](https://gradio.app/) for simplifying UI development.
