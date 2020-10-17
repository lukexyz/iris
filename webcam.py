import streamlit as st
import codecs
import streamlit.components.v1 as stc 

# Custom component
def html_component(path, width=500, height=500):
    """ 
    Custom component to open hmtl with codecs util
            path = "webcam2.html"
    """
    calc_file = codecs.open(path, 'r')
    page = calc_file.read()
    stc.html(page, width=width, height=height, scrolling=False)

def main():
    st.beta_set_page_config(
        page_title="Video capture",    # String or None. Strings get appended with "• Streamlit". 
        page_icon="📼",                # String, anything supported by st.image, or None.
        layout="centered",             # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto")  # Can be "auto", "expanded", "collapsed"

    # ================== Using st.beta_columns ================== #
    col1, col2 = st.beta_columns([4, 1]) # first column 4x the size of second

    with col2: 
        st.button("refresh")

    with col1:
        st.header("📺 Video Stream")
        st.text('Jeremy Ellis - Webcam capture on Codepen\nhttps://codepen.io/rocksetta/pen/BPbaxQ')
        st.text('Streamlit html component below')
    html_component(path="webcam2.html", width=600, height=600)


if __name__ == '__main__':
    main()
