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
    st.session_state.default_background = "https://raw.githubusercontent.com/your-github-repo/default_image.png"
if 'overlay_images' not in st.session_state:
    st.session_state.overlay_images = []
if 'texts' not in st.session_state:
    st.session_state.texts = []
if 'history' not in st.session_state:
    st.session_state.history = []

# Load default background image
if st.session_state.background_image is None:
    st.session_state.background_image = Image.open(io.BytesIO(
        st.experimental_get_binary_file(st.session_state.default_background))).convert("RGBA")

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
def default_thumbnail_layout(x_positions, y_positions, text=None):
    bg = st.session_state.background_image.copy()
    draw = ImageDraw.Draw(bg)

    for i, overlay in enumerate(st.session_state.overlay_images):
        if i < len(x_positions):
            resized_overlay = overlay['image'].resize((
                int(overlay['image'].width * overlay['scale']),
                int(overlay['image'].height * overlay['scale'])
            ))
            bg.paste(resized_overlay, (x_positions[i], y_positions[i]), resized_overlay)

    if text:
        font = ImageFont.truetype("arial.ttf", 30) if os.path.exists("arial.ttf") else ImageFont.load_default()
        draw.text((150, 30), text, fill="black", font=font)

    return bg

layouts = [
    ("Layout 1", [50, 300], [50, 300]),
    ("Layout 2", [100, 400], [50, 400]),
    ("Layout 3", [150, 500], [100, 300]),
    ("Layout 4", [200, 200], [200, 400])
]

selected_layout = None
for idx, (name, x_positions, y_positions) in enumerate(layouts):
    with columns[idx]:
        thumbnail = default_thumbnail_layout(x_positions, y_positions, text=name)
        if st.button(name):
            selected_layout = idx
        st.image(thumbnail, use_column_width=True)

# If the user selects a layout, update positions and skip canvas setup
if selected_layout is not None:
    name, x_positions, y_positions = layouts[selected_layout]
    for i, overlay in enumerate(st.session_state.overlay_images):
        if i < len(x_positions):
            overlay['x'], overlay['y'] = x_positions[i], y_positions[i]

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
canvas = st.canvas(
    background_image=st.session_state.background_image,
    width=st.session_state.background_image.width,
    height=st.session_state.background_image.height,
    drawing_mode="transform",
    update_streamlit=True
)

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
