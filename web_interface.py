import streamlit as st 
from streamlit_image_select import image_select
from streamlit_image_comparison import image_comparison
from matting import *

def main():
    # Set title for the App
    st.markdown("## :blue[U-Net: Change Portrait Background]")
    st.markdown("##### Pick an image to change the background")
#     st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #adb2ba;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
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
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)           
            _, encoded_res = cv2.imencode('.jpg', res)
            res = encoded_res.tobytes()
            st.download_button("Download Result",data=res,file_name="result.jpg",mime="image/jpg")
if __name__ == "__main__":
    main()