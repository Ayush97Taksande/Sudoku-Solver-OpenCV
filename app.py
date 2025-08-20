import streamlit as st
import numpy as np
from PIL import Image
from skimage import color, filters, measure, transform, img_as_ubyte

from Functions import *
from Sudoku_Solver import *
from Sudoku_Init import *
from PreProcess import *  # Needs to be rewritten if using cv2 inside

heightImg, widthImg = 450, 450
model = intializePredectionModel()

st.set_page_config(layout="wide")
st.title("Sudoku Solver")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Sudoku Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.success("Image uploaded successfully!")
    else:
        st.info("Please upload a Sudoku puzzle image")

col1, col2 = st.columns(2)

if uploaded_file:
    # Load and resize
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_pil = img_pil.resize((widthImg, heightImg))
    img_np = np.array(img_pil)

    with col1:
        st.header("Original Sudoku")
        st.image(img_pil, caption="Uploaded Sudoku")

    # Preprocess (threshold)
    gray = color.rgb2gray(img_np)
    thresh = gray < filters.threshold_otsu(gray)

    # Find contours
    contours = measure.find_contours(thresh.astype(float), 0.5)
    biggest = max(contours, key=len) if contours else None

    if biggest is not None:
        # Approximate bounding box for perspective transform
        minr, minc = np.min(biggest, axis=0)
        maxr, maxc = np.max(biggest, axis=0)
        src = np.array([[minc, minr], [maxc, minr], [minc, maxr], [maxc, maxr]])
        dst = np.array([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])

        tform = transform.ProjectiveTransform()
        tform.estimate(dst, src)
        warped = transform.warp(img_np, tform, output_shape=(heightImg, widthImg))
        warped_gray = color.rgb2gray(warped)

        # Split into boxes and predict
        boxes = splitBoxes((warped_gray * 255).astype(np.uint8))
        numbers = getPredection(boxes, model)
        posArray = np.where(np.array(numbers) > 0, 0, 1)

        # Solve
        board = np.array_split(numbers, 9)
        try:
            solve(board)
        except:
            pass

        solved_flat = [num for row in board for num in row]
        solvedNumbers = solved_flat * posArray

        solved_img = displayNumbers(np.zeros_like(img_np), solvedNumbers)

        # Warp back
        tform_inv = transform.ProjectiveTransform()
        tform_inv.estimate(src, dst)
        inv_warp = transform.warp(solved_img, tform_inv, output_shape=(heightImg, widthImg))

        # Blend images
        blended = np.clip(inv_warp * 255 + img_np * 0.5, 0, 255).astype(np.uint8)

        with col2:
            st.header("Solved Sudoku")
            st.image(blended, caption="Solved Sudoku")
            st.download_button(
                "Download Solved Image",
                data=Image.fromarray(blended).tobytes(),
                file_name="solved_sudoku.png"
            )
    else:
        st.error("Could not detect Sudoku grid")