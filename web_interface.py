import streamlit as st 
from streamlit_image_select import image_select
from streamlit_image_comparison import image_comparison
from matting import *

def main():
    # Set title for the App
    st.markdown("## :blue[U-Net: Change Portrait Background]")
    st.markdown("##### Pick an image to change the background")
    
    # Draw a dividing line
    st.divider()
    
    u_net = load_model()
    
    uploaded_file = st.file_uploader(" #### :camera: :violet[1. Upload Portrait Image] ", type=["jpg", "JPEG", "png"])
    if uploaded_file is not None:
        # Display uploaded image with label
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
               
        # Load image and mask
        img = load_img(str(uploaded_file.name))
        #mask = load_img('mask.png',True)
        
        # Draw a dividing line
        st.divider()
        st.markdown("#### :violet[2. Portrait Mask]")
        mask = create_mask(img, u_net)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)    
        st.image(mask)
        # Select available background
        bg = image_select(
            label="Select a background",
            images=[
                "image/bg.jpg",
                "image/desert.jpg",
                "image/moon.jpg",
                "image/beach.jpg"
            ]          
        )
        bg = load_img(bg)
        click = st.button("Change background")
        if(click):
            res = change_pixel(img, bg, mask)
            st.divider()
            # Config a slider to compare the original portrait vs result 
            image_comparison(
                img1= res,
                img2= img
            )
        
if __name__ == "__main__":
    main()