import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rembg import remove
import io
import os
import base64
from streamlit_drawable_canvas import st_canvas

# Function to convert PIL image to base64
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

# Set page configuration
st.set_page_config(page_title="Thumbnail Editor", page_icon="🎨", layout="wide")

# Initialize session state
if 'background_image' not in st.session_state:
    st.session_state.background_image = None
if 'default_background' not in st.session_state:
    st.session_state.default_background = "youtube-thumbnail-orange-gradient-nm9iw60na2j0ibcy.jpg"
if 'overlay_images' not in st.session_state:
    st.session_state.overlay_images = []
if 'texts' not in st.session_state:
    st.session_state.texts = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_layout' not in st.session_state:
    st.session_state.selected_layout = None

# Load default background image
if st.session_state.background_image is None:
    st.session_state.background_image = Image.open(st.session_state.default_background).convert("RGBA")

# Sidebar components
st.sidebar.title("Thumbnail Editor Options")

# Upload background image
st.sidebar.write("### Upload Background Image")
background_file = st.sidebar.file_uploader(
    "Drag and drop or click to upload your background image.",
    type=["png", "jpg", "jpeg"]
)
if background_file:
    st.session_state.background_image = Image.open(background_file).convert("RGBA")
    st.session_state.history.append({'background_image': st.session_state.background_image.copy()})

# Sidebar: Upload Overlay Images
st.sidebar.write("### Upload Overlay Images")
overlay_files = st.sidebar.file_uploader(
    "Drag and drop or click to upload overlay images.",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)
for overlay_file in overlay_files:
    overlay_bytes = remove(overlay_file.read())
    overlay_image = Image.open(io.BytesIO(overlay_bytes)).convert("RGBA")
    st.session_state.overlay_images.append({'image': overlay_image, 'x': 50, 'y': 50, 'scale': 1.0})

# Interactive Canvas
st.write("## Customize Your Thumbnail")
background_image_url = (
    image_to_base64(st.session_state.background_image) if isinstance(st.session_state.background_image, Image.Image) else None
)

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=3,
    stroke_color="#000000",
    background_image=st.session_state.background_image if isinstance(st.session_state.background_image, Image.Image) else None,
    update_streamlit=True,
    height=st.session_state.background_image.height if isinstance(st.session_state.background_image, Image.Image) else 400,
    width=st.session_state.background_image.width if isinstance(st.session_state.background_image, Image.Image) else 600,
    drawing_mode="transform",
    display_toolbar=True,
    point_display_radius=0,
    key="canvas",
)

# Manage overlay images and text dynamically
if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    for obj in objects:
        if obj.get("type") == "image":
            st.session_state.overlay_images.append({
                'image': Image.open(io.BytesIO(base64.b64decode(obj["src"].split(",")[-1]))),
                'x': obj["left"],
                'y': obj["top"],
                'scale': obj["scaleX"]
            })
        elif obj.get("type") == "textbox":
            st.session_state.texts.append({
                'content': obj["text"],
                'x': obj["left"],
                'y': obj["top"],
                'color': obj["fill"],
                'size': obj["fontSize"]
            })

# Save Button
if st.sidebar.button("Save Thumbnail"):
    buffer = io.BytesIO()
    bg_image = st.session_state.background_image.copy()
    draw = ImageDraw.Draw(bg_image)

    for overlay in st.session_state.overlay_images:
        resized_overlay = overlay['image'].resize((
            int(overlay['image'].width * overlay['scale']),
            int(overlay['image'].height * overlay['scale'])
        ))
        bg_image.paste(resized_overlay, (int(overlay['x']), int(overlay['y'])), resized_overlay)

    for text in st.session_state.texts:
        font = ImageFont.truetype("arial.ttf", int(text['size'])) if os.path.exists("arial.ttf") else ImageFont.load_default()
        draw.text((int(text['x']), int(text['y'])), text['content'], fill=text['color'], font=font)

    bg_image.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button("Download Thumbnail", buffer, file_name="thumbnail.png", mime="image/png")
