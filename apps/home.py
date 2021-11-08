import streamlit as st
import base64

def app():
    main_bg = "bg-6-dd.jpg"
    main_bg_ext = "jpg"
    
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.write('')
#     st.title('Home')

#     st.write('This is the `home page` of this multi-page app.')

#     st.write('In this app, we will be building a simple classification model using the Iris dataset.')
