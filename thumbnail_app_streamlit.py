import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
import io
import os
from rembg import remove
from streamlit_drawable_canvas import st_canvas

# Set page configuration
st.set_page_config(page_title="Thumbnail Editor", page_icon="🎨", layout="wide")

# Initialize session state
if "background_image" not in st.session_state:
    st.session_state.background_image = None
if "overlay_images" not in st.session_state:
    st.session_state.overlay_images = []
if "texts" not in st.session_state:
    st.session_state.texts = []

# Sidebar: Background Image
st.sidebar.title("Thumbnail Editor Options")
st.sidebar.header("Background Image")
background_file = st.sidebar.file_uploader("Upload a background image", type=["png", "jpg", "jpeg"])
if background_file:
    st.session_state.background_image = Image.open(background_file).convert("RGBA")
else:
    st.session_state.background_image = Image.new("RGBA", (800, 600), "white")

# Sidebar: Overlay Images
st.sidebar.header("Overlay Images")
overlay_files = st.sidebar.file_uploader("Upload overlay images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if overlay_files:
    for file in overlay_files:
        image_data = remove(file.read())
        overlay_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        st.session_state.overlay_images.append({"image": overlay_image, "x": 50, "y": 50, "scale": 1.0})

# Sidebar: Text Options
st.sidebar.header("Text Options")
if st.sidebar.button("Add Text"):
    st.session_state.texts.append({"content": "New Text", "x": 50, "y": 50, "color": "black", "size": 24})

# Drawable Canvas
st.write("### Drawable Canvas")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Transparent fill color
    stroke_width=3,
    stroke_color="#000000",
    background_image=st.session_state.background_image if isinstance(st.session_state.background_image, Image.Image) else None,
    update_streamlit=True,
    height=st.session_state.background_image.height,
    width=st.session_state.background_image.width,
    drawing_mode="transform",
    display_toolbar=True,
    key="canvas",
)

# Render overlay images dynamically
if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    for obj in objects:
        if obj.get("type") == "image":
            st.session_state.overlay_images.append({
                "image": Image.open(io.BytesIO(base64.b64decode(obj["src"].split(",")[-1]))),
                "x": obj["left"],
                "y": obj["top"],
                "scale": obj["scaleX"]
            })
        elif obj.get("type") == "textbox":
            st.session_state.texts.append({
                "content": obj["text"],
                "x": obj["left"],
                "y": obj["top"],
                "color": obj["fill"],
                "size": obj["fontSize"]
            })

# Save the final image
if st.sidebar.button("Save Thumbnail"):
    buffer = io.BytesIO()
    bg_image = st.session_state.background_image.copy()
    draw = ImageDraw.Draw(bg_image)

    # Add overlay images
    for overlay in st.session_state.overlay_images:
        resized_overlay = overlay["image"].resize((
            int(overlay["image"].width * overlay["scale"]),
            int(overlay["image"].height * overlay["scale"])
        ))
        bg_image.paste(resized_overlay, (int(overlay["x"]), int(overlay["y"])), resized_overlay)

    # Add text
    for text in st.session_state.texts:
        font = ImageFont.truetype("arial.ttf", int(text["size"])) if os.path.exists("arial.ttf") else ImageFont.load_default()
        draw.text((int(text["x"]), int(text["y"])), text["content"], fill=text["color"], font=font)

    bg_image.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button("Download Thumbnail", buffer, file_name="thumbnail.png", mime="image/png")
