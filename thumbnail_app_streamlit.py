import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rembg import remove
import io
import os
import json

# Set page configuration
st.set_page_config(page_title="Thumbnail Editor", page_icon="🎨", layout="wide")

# Initialize session state
if 'background_image' not in st.session_state:
    st.session_state.background_image = None
if 'overlay_images' not in st.session_state:
    st.session_state.overlay_images = []
if 'texts' not in st.session_state:
    st.session_state.texts = []
if 'history' not in st.session_state:
    st.session_state.history = []

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
else:
    st.sidebar.info("Drag and drop an image file here to set your background.")

# Upload overlay images
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

# Text management
st.sidebar.write("### Add and Edit Text")
if st.sidebar.button("Add Text"):
    st.session_state.texts.append({'content': "Your Text Here", 'x': 100, 'y': 100, 'color': "#000000", 'size': 20})

for i, text in enumerate(st.session_state.texts):
    st.sidebar.write(f"#### Text {i + 1}")
    text['content'] = st.sidebar.text_input(f"Text Content {i + 1}", text['content'])
    text['color'] = st.sidebar.color_picker(f"Text Color {i + 1}", text['color'])
    text['size'] = st.sidebar.slider(f"Font Size {i + 1}", 10, 100, text['size'])
    text['x'] = st.sidebar.slider(f"Text X Position {i + 1}", 0, 1024, text['x'])
    text['y'] = st.sidebar.slider(f"Text Y Position {i + 1}", 0, 1024, text['y'])

# Save button
save_button = st.sidebar.button("Save Thumbnail")
undo_button = st.sidebar.button("Undo")
if undo_button and st.session_state.history:
    last_state = st.session_state.history.pop()
    st.session_state.background_image = last_state.get('background_image', st.session_state.background_image)
    st.session_state.overlay_images = last_state.get('overlay_images', st.session_state.overlay_images)
    st.session_state.texts = last_state.get('texts', st.session_state.texts)

# Main content area
st.title("Thumbnail Editor")
if st.session_state.background_image:
    bg_image = st.session_state.background_image.copy()
    draw = ImageDraw.Draw(bg_image)

    # Render overlay images
    for overlay in st.session_state.overlay_images:
        overlay_image = overlay['image'].resize((
            int(overlay['image'].width * overlay['scale']),
            int(overlay['image'].height * overlay['scale'])
        ))
        bg_image.paste(overlay_image, (overlay['x'], overlay['y']), overlay_image)

    # Render text
    for text in st.session_state.texts:
        font = ImageFont.truetype("arial.ttf", text['size']) if os.path.exists("arial.ttf") else ImageFont.load_default()
        draw.text((text['x'], text['y']), text['content'], fill=text['color'], font=font)

    # Display the final image
    st.image(bg_image, caption="Thumbnail Preview", use_column_width=True)

    # Save the thumbnail
    if save_button:
        buffer = io.BytesIO()
        bg_image.save(buffer, format="PNG")
        buffer.seek(0)
        st.download_button(
            label="Download Thumbnail",
            data=buffer,
            file_name="thumbnail.png",
            mime="image/png"
        )
else:
    st.write("Upload a background image to get started.")
