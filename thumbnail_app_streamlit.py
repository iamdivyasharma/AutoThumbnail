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
if 'overlay_image' not in st.session_state:
    st.session_state.overlay_image = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar components
st.sidebar.title("Thumbnail Editor Options")

# Upload background image with placeholder
background_file = st.sidebar.file_uploader(
    "Upload Background Image", 
    type=["png", "jpg", "jpeg"],
    help="Upload the main image that will serve as the thumbnail background."
)
if background_file:
    st.session_state.background_image = Image.open(background_file)
    st.session_state.history.append(st.session_state.background_image.copy())
else:
    st.sidebar.write("No background image uploaded yet.")

# Upload overlay image with placeholder
overlay_file = st.sidebar.file_uploader(
    "Upload Overlay Image", 
    type=["png", "jpg", "jpeg"],
    help="Upload an image to overlay on the background (e.g., logos or decorations)."
)
if overlay_file:
    # Remove background from overlay image
    overlay_bytes = remove(overlay_file.read())
    overlay_image = Image.open(io.BytesIO(overlay_bytes)).convert("RGBA")
    st.session_state.overlay_image = overlay_image
else:
    st.sidebar.write("No overlay image uploaded yet.")

# Text options
text_content = st.sidebar.text_input("Add Text", "Your Text Here")
text_color = st.sidebar.color_picker("Pick Text Color", "#000000")
font_size = st.sidebar.slider("Font Size", 10, 100, 20)
text_x = st.sidebar.slider("Text X Position", 0, 1024, 100)
text_y = st.sidebar.slider("Text Y Position", 0, 1024, 100)

# Overlay transformations
scale = st.sidebar.slider("Overlay Scale", 0.5, 3.0, 1.0)
overlay_x = st.sidebar.slider("Overlay X Position", 0, 1024, 50)
overlay_y = st.sidebar.slider("Overlay Y Position", 0, 1024, 50)
action = st.sidebar.selectbox("Transformation", ["None", "Flip Horizontal", "Flip Vertical", "Mirror"])

# Bounding box options
show_bounding_box = st.sidebar.checkbox("Show Bounding Box", value=True)
bounding_box_style = st.sidebar.selectbox("Bounding Box Style", ["Rectangle", "Cloud"])
bounding_box_color = st.sidebar.color_picker("Bounding Box Color", "#FF0000")
bounding_box_fill = st.sidebar.color_picker("Bounding Box Fill", "#FFFFFF")

# Font upload
default_font_path = "arial.ttf"
font_file = st.sidebar.file_uploader("Upload Font", type=["ttf"])
if font_file:
    font_path = font_file.name
    with open(font_path, "wb") as f:
        f.write(font_file.read())
else:
    font_path = default_font_path

# Save button
save_button = st.sidebar.button("Save Thumbnail")
undo_button = st.sidebar.button("Undo")
if undo_button and st.session_state.history:
    st.session_state.background_image = st.session_state.history.pop()

# Main content area
st.title("Thumbnail Editor")

if st.session_state.background_image:
    bg_image = st.session_state.background_image.copy()

    # Resize overlay and apply transformations
    if st.session_state.overlay_image:
        overlay_image = st.session_state.overlay_image.copy()
        overlay_image = overlay_image.resize((
            int(overlay_image.width * scale),
            int(overlay_image.height * scale)
        ))

        if action == "Flip Horizontal":
            overlay_image = overlay_image.transpose(Image.FLIP_LEFT_RIGHT)
        elif action == "Flip Vertical":
            overlay_image = overlay_image.transpose(Image.FLIP_TOP_BOTTOM)
        elif action == "Mirror":
            overlay_image = ImageOps.mirror(overlay_image)

        # Paste overlay onto background
        bg_image.paste(overlay_image, (overlay_x, overlay_y), overlay_image)

    # Add text to the image
    draw = ImageDraw.Draw(bg_image)
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    text_position = (text_x, text_y)
    draw.text(text_position, text_content, fill=text_color, font=font)

    # Draw bounding box if enabled
    if show_bounding_box:
        text_width, text_height = draw.textsize(text_content, font=font)
        x, y = text_position
        if bounding_box_style == "Rectangle":
            draw.rectangle([
                (x - 5, y - 5),
                (x + text_width + 5, y + text_height + 5)
            ], fill=bounding_box_fill, outline=bounding_box_color, width=2)
        elif bounding_box_style == "Cloud":
            draw.ellipse([
                (x - 10, y - 10),
                (x + text_width + 10, y + text_height + 10)
            ], fill=bounding_box_fill, outline=bounding_box_color, width=2)

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
