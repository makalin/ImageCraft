import gradio as gr
import cv2
import numpy as np
from datetime import datetime
import random

def basic_filters(image, filter_type):
    """Applies basic image filters"""
    if filter_type == "Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == "Sepia":
        sepia_filter = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        return cv2.transform(image, sepia_filter)
    elif filter_type == "X-Ray":
        # Enhanced X-ray effect
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(inverted)
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Blur":
        return cv2.GaussianBlur(image, (15, 15), 0)

def classic_filters(image, filter_type):
    """Classic image filters"""
    if filter_type == "Pencil Sketch":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        sketch = cv2.divide(gray, cv2.subtract(255, blurred), scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == "Sharpen":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    elif filter_type == "Emboss":
        kernel = np.array([[0,-1,-1], [1,0,-1], [1,1,0]])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        emboss = cv2.filter2D(gray, -1, kernel) + 128
        return cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == "Edge Detection":
        edges = cv2.Canny(image, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def creative_filters(image, filter_type):
    """Creative and unusual image filters"""
    if filter_type == "Pixel Art":
        h, w = image.shape[:2]
        pixel_size = 20
        small = cv2.resize(image, (w//pixel_size, h//pixel_size))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    elif filter_type == "Mosaic Effect":
        h, w = image.shape[:2]
        mosaic_size = 30
        for i in range(0, h, mosaic_size):
            for j in range(0, w, mosaic_size):
                roi = image[i:i+mosaic_size, j:j+mosaic_size]
                if roi.size > 0:
                    color = np.mean(roi, axis=(0,1))
                    image[i:i+mosaic_size, j:j+mosaic_size] = color
        return image
    
    elif filter_type == "Rainbow":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        for i in range(h):
            hsv[i, :, 0] = (hsv[i, :, 0] + i % 180).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif filter_type == "Night Vision":
        green_image = image.copy()
        green_image[:,:,0] = 0  # Blue channel
        green_image[:,:,2] = 0  # Red channel
        return cv2.addWeighted(green_image, 1.5, np.zeros(image.shape, image.dtype), 0, -50)

def special_effects(image, filter_type):
    """Applies special effects"""
    if filter_type == "Matrix Effect":
        green_matrix = np.zeros_like(image)
        green_matrix[:,:,1] = image[:,:,1]  # Only green channel
        random_brightness = np.random.randint(0, 255, size=image.shape[:2])
        green_matrix[:,:,1] = np.minimum(green_matrix[:,:,1] + random_brightness, 255)
        return green_matrix
    
    elif filter_type == "Wave Effect":
        rows, cols = image.shape[:2]
        img_output = np.zeros(image.shape, dtype=image.dtype)
        
        for i in range(rows):
            for j in range(cols):
                offset_x = int(25.0 * np.sin(2 * 3.14 * i / 180))
                offset_y = int(25.0 * np.cos(2 * 3.14 * j / 180))
                if i+offset_x < rows and j+offset_y < cols:
                    img_output[i,j] = image[(i+offset_x)%rows,(j+offset_y)%cols]
                else:
                    img_output[i,j] = 0
        return img_output
    
    elif filter_type == "Timestamp":
        output = image.copy()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, timestamp, (10, 30), font, 1, (255, 255, 255), 2)
        return output
    
    elif filter_type == "Glitch Effect":
        glitch = image.copy()
        h, w = image.shape[:2]
        for _ in range(10):
            x1 = random.randint(0, w-50)
            y1 = random.randint(0, h-50)
            x2 = random.randint(x1, min(x1+50, w))
            y2 = random.randint(y1, min(y1+50, h))
            glitch[y1:y2, x1:x2] = np.roll(glitch[y1:y2, x1:x2], 
                                          random.randint(-20, 20), 
                                          axis=random.randint(0, 1))
        return glitch

def artistic_filters(image, filter_type):
    """Applies artistic image filters"""
    if filter_type == "Pop Art":
        img_small = cv2.resize(image, None, fx=0.5, fy=0.5)
        img_color = cv2.resize(img_small, (image.shape[1], image.shape[0]))
        for _ in range(2):
            img_color = cv2.bilateralFilter(img_color, 9, 300, 300)
        hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = hsv[:,:,1]*1.5
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif filter_type == "Oil Paint":
        ret = np.float32(image.copy())
        ret = cv2.bilateralFilter(ret, 9, 75, 75)
        ret = cv2.detailEnhance(ret, sigma_s=15, sigma_r=0.15)
        ret = cv2.edgePreservingFilter(ret, flags=1, sigma_s=60, sigma_r=0.4)
        return np.uint8(ret)
    
    elif filter_type == "Cartoon":
        # Enhanced cartoon effect
        color = image.copy()
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(color, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        # Increase color saturation
        hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = hsv[:,:,1]*1.4  # Increase saturation
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def atmospheric_filters(image, filter_type):
    """Applies atmospheric filters"""
    if filter_type == "Autumn":
        autumn_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        autumn = cv2.transform(image, autumn_filter)
        # Increase color warmth
        hsv = cv2.cvtColor(autumn, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = hsv[:,:,0]*0.8  # Shift towards orange/yellow tones
        hsv[:,:,1] = hsv[:,:,1]*1.2  # Increase saturation
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif filter_type == "Nostalgia":
        image = cv2.convertScaleAbs(image, alpha=0.9, beta=10)
        sepia = cv2.transform(image, np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ]))
        # Add vignette effect
        h, w = image.shape[:2]
        kernel = np.zeros((h, w))
        center = (h//2, w//2)
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2)
                kernel[i,j] = 1 - min(1, dist/(np.sqrt(h**2 + w**2)/2))
        kernel = np.dstack([kernel]*3)
        return cv2.multiply(sepia, kernel).astype(np.uint8)
    
    elif filter_type == "Brightness Increase":
        # Enhanced brightness increase
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Increase brightness
        hsv[:,:,2] = cv2.convertScaleAbs(hsv[:,:,2], alpha=1.2, beta=30)
        # Slightly increase contrast
        return cv2.convertScaleAbs(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), alpha=1.1, beta=0)

basic_filters_list = ["Grayscale", "Sepia", "X-Ray", "Blur"]
classic_filters_list = ["Pencil Sketch", "Sharpen", "Emboss", "Edge Detection"]
creative_filters_list = ["Pixel Art", "Mosaic Effect", "Rainbow", "Night Vision"]
special_effects_list = ["Matrix Effect", "Wave Effect", "Timestamp", "Glitch Effect"]
artistic_filters_list = ["Pop Art", "Oil Paint", "Cartoon"]
atmospheric_filters_list = ["Autumn", "Nostalgia", "Brightness Increase"]

def image_processing(image, filters):
    """Main image processing function"""
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for filter_type in filters:
        if filter_type in basic_filters_list:
            image = basic_filters(image, filter_type)
        elif filter_type in classic_filters_list:
            image = classic_filters(image, filter_type)
        elif filter_type in creative_filters_list:
            image = creative_filters(image, filter_type)
        elif filter_type in special_effects_list:
            image = special_effects(image, filter_type)
        elif filter_type in artistic_filters_list:
            image = artistic_filters(image, filter_type)
        elif filter_type in atmospheric_filters_list:
            image = atmospheric_filters(image, filter_type)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image

with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# 🎨 Image Filter Studio")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="📸 Upload Photo")
            with gr.Accordion("ℹ️ Filter Categories", open=True):
                filters = gr.CheckboxGroup(
                    [
                        "Grayscale", "Sepia", "X-Ray", "Blur",
                        "Pencil Sketch", "Sharpen", "Emboss", "Edge Detection",
                        "Pixel Art", "Mosaic Effect", "Rainbow", "Night Vision",
                        "Matrix Effect", "Wave Effect", "Timestamp", "Glitch Effect",
                        "Pop Art", "Oil Paint", "Cartoon",
                        "Autumn", "Nostalgia", "Brightness Increase"
                    ],
                    label="🎭 Choose Filter(s)",
                    info="Select multiple effect to apply"
                )
            submit_button = gr.Button("✨ Apply Filter(s)", variant="primary")
            
        with gr.Column():
            image_output = gr.Image(label="🖼️ Filtered Photo")
    
    submit_button.click(
        image_processing,
        inputs=[image_input, filters],
        outputs=image_output
    )

app.launch(share=True)
