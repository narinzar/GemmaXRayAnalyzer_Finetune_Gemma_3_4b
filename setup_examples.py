# setup_examples.py
# Purpose: Create the examples directory and download sample X-ray images

import os
import urllib.request
from PIL import Image
import io

# Create examples directory if it doesn't exist
if not os.path.exists("examples"):
    os.makedirs("examples")
    print("Created examples directory")

# URLs for sample X-ray images
# These are example URLs for demonstration - replace with actual URLs in production
SAMPLE_IMAGES = {
    "pneumonia": "https://prod-images-static.radiopaedia.org/images/53748657/332aa629dbe4be3eaea8d776a4cf7a_big_gallery.jpeg",
    "normal": "https://prod-images-static.radiopaedia.org/images/70242828/eb5fe57f81de1b2ac000e8b06911ce_big_gallery.jpeg"
}

# Download and save sample images
for name, url in SAMPLE_IMAGES.items():
    try:
        # Create a custom opener to handle HTTP redirects
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        # Download image
        with urllib.request.urlopen(url) as response:
            image_data = response.read()
        
        # Open as PIL Image and save
        image = Image.open(io.BytesIO(image_data))
        save_path = os.path.join("examples", f"{name}.jpg")
        image.save(save_path)
        print(f"Downloaded and saved {save_path}")
    
    except Exception as e:
        print(f"Error downloading {name} image: {e}")
        # Create a placeholder image instead
        placeholder = Image.new('RGB', (512, 512), color=(240, 240, 240))
        save_path = os.path.join("examples", f"{name}.jpg")
        placeholder.save(save_path)
        print(f"Created placeholder image at {save_path}")

print("Example setup complete")
