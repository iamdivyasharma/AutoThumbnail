import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rembg import remove
import io
import os
from streamlit_drawable_canvas import st_canvas

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
if 'selected_layout' not in st.session_state:
    st.session_state.selected_layout = None

# Load default background image
if st.session_state.background_image is None:
    st.session_state.background_image = Image.open(st.session_state.default_background).convert("RGBA")

# Sidebar: Upload background image
st.sidebar.write("### Upload Background Image")
background_file = st.sidebar.file_uploader("Upload a background image.", type=["png", "jpg", "jpeg"])
if background_file:
    st.session_state.background_image = Image.open(background_file)

# Sidebar: Upload overlay images
st.sidebar.write("### Upload Overlay Images")
overlay_files = st.sidebar.file_uploader("Upload overlay images.", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
for overlay_file in overlay_files:
    overlay_bytes = remove(overlay_file.read())
    overlay_image = Image.open(io.BytesIO(overlay_bytes)).convert("RGBA")
    st.session_state.overlay_images.append({'image': overlay_image, 'x': 50, 'y': 50, 'scale': 1.0})

# Sidebar: Add text boxes
if st.sidebar.button("Add Text Box"):
    st.session_state.texts.append({'content': "Your Text Here", 'x': 100, 'y': 100, 'color': "#000000", 'size': 20})

# Default layouts
st.write("## Choose a Default Thumbnail Layout")
columns = st.columns(4)

layouts = [
    ("Layout 1", [(50, 50), (300, 300)]),
    ("Layout 2", [(50, 300), (300, 50)]),
    ("Layout 3", [(100, 100), (400, 400)]),
    ("Layout 4 (Mirrored)", [(50, 50), (300, 300)], True)
]

for idx, layout in enumerate(layouts):
    layout_name, positions = layout[:2]
    flip_images = layout[2] if len(layout) > 2 else False

    def apply_layout():
        st.session_state.selected_layout = layout
        for i, overlay in enumerate(st.session_state.overlay_images):
            if i < len(positions):
                overlay['x'], overlay['y'] = positions[i]
                if flip_images and i % 2 == 1:
                    overlay['image'] = ImageOps.mirror(overlay['image'])

    with columns[idx]:
        st.button(layout_name, on_click=apply_layout)

# Canvas for customization
st.write("## Customize Your Thumbnail")
canvas_result = st_canvas(
    background_image=st.session_state.background_image,
    width=st.session_state.background_image.width,
    height=st.session_state.background_image.height,
    drawing_mode="transform",
    update_streamlit=True,
    key="canvas",
)

# Save button
if st.sidebar.button("Save Thumbnail"):
    buffer = io.BytesIO()
    bg_image = st.session_state.background_image.copy()
    draw = ImageDraw.Draw(bg_image)

    # Draw overlays
    for overlay in st.session_state.overlay_images:
        resized_overlay = overlay['image'].resize((
            int(overlay['image'].width * overlay['scale']),
            int(overlay['image'].height * overlay['scale'])
        ))
        bg_image.paste(resized_overlay, (overlay['x'], overlay['y']), resized_overlay)

    # Draw texts
    for text in st.session_state.texts:
        font = ImageFont.truetype("arial.ttf", text['size']) if os.path.exists("arial.ttf") else ImageFont.load_default()
        draw.text((text['x'], text['y']), text['content'], fill=text['color'], font=font)

    bg_image.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button("Download Thumbnail", buffer, file_name="thumbnail.png", mime="image/png")
