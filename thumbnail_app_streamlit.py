import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rembg import remove
import io
import os

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
    st.session_state.background_image = Image.open(background_file)
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

# Generate Default Thumbnails
st.write("## Choose a Default Thumbnail Layout")
columns = st.columns(4)
def default_thumbnail_layout(layout_name, positions, flip_images=False):
    bg = st.session_state.background_image.copy()
    draw = ImageDraw.Draw(bg)

    for i, overlay in enumerate(st.session_state.overlay_images):
        if i < len(positions):
            resized_overlay = overlay['image'].resize((
                int(overlay['image'].width * overlay['scale']),
                int(overlay['image'].height * overlay['scale'])
            ))

            if flip_images and i % 2 == 1:  # Flip alternate images
                resized_overlay = ImageOps.mirror(resized_overlay)

            x, y = positions[i]
            bg.paste(resized_overlay, (x, y), resized_overlay)

    font = ImageFont.truetype("arial.ttf", 30) if os.path.exists("arial.ttf") else ImageFont.load_default()
    draw.text((20, 20), layout_name, fill="black", font=font)

    return bg

# Define layouts
layouts = [
    ("Layout 1", [(50, 50), (300, 300)]),
    ("Layout 2", [(50, 300), (300, 50)]),
    ("Layout 3", [(100, 100), (400, 400)]),
    ("Layout 4 (Mirrored)", [(50, 50), (300, 300)], True)
]

selected_layout = None
for idx, layout in enumerate(layouts):
    layout_name = layout[0]
    positions = layout[1]
    flip_images = layout[2] if len(layout) > 2 else False

    with columns[idx]:
        thumbnail = default_thumbnail_layout(layout_name, positions, flip_images=flip_images)
        if st.button(layout_name):
            selected_layout = layout
        st.image(thumbnail, use_container_width=True)

# Apply selected layout if chosen
if selected_layout is not None:
    layout_name, positions, *flip_images = selected_layout
    flip_images = flip_images[0] if flip_images else False
    for i, overlay in enumerate(st.session_state.overlay_images):
        if i < len(positions):
            overlay['x'], overlay['y'] = positions[i]
            if flip_images and i % 2 == 1:
                overlay['image'] = ImageOps.mirror(overlay['image'])

# Text management
st.sidebar.write("### Add and Edit Text")
if st.sidebar.button("Add Text Box"):
    st.session_state.texts.append({'content': "Your Text Here", 'x': 100, 'y': 100, 'color': "#000000", 'size': 20})

for i, text in enumerate(st.session_state.texts):
    st.sidebar.write(f"#### Text {i + 1}")
    text['content'] = st.sidebar.text_input(f"Text Content {i + 1}", text['content'])
    text['color'] = st.sidebar.color_picker(f"Text Color {i + 1}", text['color'])
    text['size'] = st.sidebar.slider(f"Font Size {i + 1}", 10, 100, text['size'])

# Interactive Canvas
st.write("## Customize Your Thumbnail")
# Placeholder for future canvas integration

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
        bg_image.paste(resized_overlay, (overlay['x'], overlay['y']), resized_overlay)

    for text in st.session_state.texts:
        font = ImageFont.truetype("arial.ttf", text['size']) if os.path.exists("arial.ttf") else ImageFont.load_default()
        draw.text((text['x'], text['y']), text['content'], fill=text['color'], font=font)

    bg_image.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button("Download Thumbnail", buffer, file_name="thumbnail.png", mime="image/png")
