import io
import tempfile
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# Make sure this import works with your local structure
from src.agent import SareeAgent 


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TailorTalk — AI Saree Search",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS (Theme Aware)
# ============================================================

st.markdown(
    """
    <style>
    /* -------------------------------------------------------
       Global & Hero
    ------------------------------------------------------- */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero {
        padding: 2.5rem 2.2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(255, 111, 145, 0.1) 0%, rgba(255, 150, 113, 0.05) 100%);
        border: 1px solid var(--border-color, #e4e4e7);
        margin-bottom: 2rem;
        text-align: center;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
        color: var(--text-color);
    }

    .hero-subtitle {
        color: var(--text-color);
        opacity: 0.8;
        font-size: 1.1rem;
        margin-bottom: 0;
    }

    /* -------------------------------------------------------
       Empty state
       ------------------------------------------------------- */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        border: 2px dashed var(--border-color, #d4d4d8);
        border-radius: 16px;
        background: transparent;
        margin-top: 2rem;
    }

    .empty-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }

    .empty-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-color);
    }

    .empty-text {
        color: var(--text-color);
        opacity: 0.7;
    }

    /* Price Styling */
    .price-tag {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-color);
    }
    
    .old-price {
        color: var(--text-color);
        opacity: 0.5;
        text-decoration: line-through;
        font-size: 0.9rem;
        margin-left: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AGENT
# ============================================================

@st.cache_resource
def get_agent():
    return SareeAgent()

agent = get_agent()


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state():
    defaults = {
        "query_path": None,
        "query_image": None,
        "agent_message": None,
        "messages": []
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def reset_session():
    for key in ["query_path", "query_image", "agent_message", "messages"]:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()

init_session_state()


# ============================================================
# HELPERS
# ============================================================

def save_uploaded_image(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()
    return Path(temp_file.name)

def download_image_from_url(url):
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name, format="JPEG")
    temp_file.close()
    return Path(temp_file.name), image

def get_result_image(result):
    image_url = result.get("image_url")
    if image_url:
        try:
            response = requests.get(image_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            pass

    # Local fallback
    local_path = Path("data") / "images" / result.get("image_filename", "")
    if local_path.exists():
        return Image.open(local_path).convert("RGB")

    return None

def run_visual_search(message):
    if not st.session_state.query_path:
        st.sidebar.warning("Please upload an image or provide an image URL first.")
        return

    agent.image_path = Path(st.session_state.query_path)
    with st.spinner("Finding visually similar sarees..."):
        response = agent.chat(message)

    st.session_state.agent_message = response["message"]
    st.session_state.messages.append({"role": "user", "content": message})
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response["message"],
        "results": response.get("results", [])
    })


# ============================================================
# MAIN UI
# ============================================================

# --- HERO SECTION ---
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">👗 TailorTalk</div>
        <p class="hero-subtitle">Discover visually similar sarees using AI-powered fashion search.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- SIDEBAR ---
with st.sidebar:
    st.header("🔎 Image Search")
    st.caption("Upload a saree image or paste a URL to begin.")

    input_mode = st.radio("Image source", ["Upload image", "Image URL"], horizontal=True, label_visibility="collapsed")

    # Upload Mode
    if input_mode == "Upload image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if uploaded_file:
            try:
                st.session_state.query_image = Image.open(uploaded_file).convert("RGB")
                st.session_state.query_path = save_uploaded_image(uploaded_file)
            except Exception as e:
                st.error(f"Could not read image: {e}")

    # URL Mode
    else:
        image_url = st.text_input("Image URL", placeholder="https://example.com/saree.jpg")
        if st.button("Load Image", use_container_width=True):
            if not image_url.strip():
                st.warning("Please enter an image URL.")
            else:
                try:
                    with st.spinner("Loading image..."):
                        path, image = download_image_from_url(image_url.strip())
                    st.session_state.query_path = path
                    st.session_state.query_image = image
                except Exception as e:
                    st.error("Could not load the image. Ensure the URL points directly to an image file.")

    # Image Preview & Quick Actions
    if st.session_state.query_image:
        st.divider()
        st.markdown("**Your Query Image**")
        st.image(st.session_state.query_image, use_container_width=True, caption="Target Saree")
        
        if st.button("⚡ Find Exact Matches", use_container_width=True, type="primary"):
            run_visual_search("Find visually similar sarees to this image.")
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Session", use_container_width=True):
        reset_session()
        st.rerun()


# --- CHAT INTERFACE ---
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">✨</div>
            <div class="empty-title">Ready to discover your next saree?</div>
            <div class="empty-text">Upload an image in the sidebar, then ask me to find similar colors, patterns, or styles.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If the assistant message contains results, render the visual grid inside the chat!
            if message.get("results"):
                st.markdown("<br>", unsafe_allow_html=True) # Add a little breathing room
                
                results = message["results"]
                # 3 columns work best inside the slightly narrower chat container
                for start in range(0, len(results), 3):
                    row = results[start:start + 3]
                    cols = st.columns(3, gap="medium")

                    for idx, (col, result) in enumerate(zip(cols, row), start=start + 1):
                        with col:
                            with st.container(border=True):
                                # Image
                                image = get_result_image(result)
                                if image:
                                    st.image(image, use_container_width=True)
                                else:
                                    st.image("https://via.placeholder.com/400x500?text=Image+Unavailable", use_container_width=True)
                                
                                # Metadata
                                st.caption(f"#{idx} • {result.get('score', 0) * 100:.1f}% Match")
                                st.markdown(f"**{result.get('name', 'Unknown Saree')}**")
                                
                                # Pricing logic
                                retail = result.get("retail_price")
                                discounted = result.get("discounted_price")
                                
                                if discounted is not None and retail is not None and discounted != retail:
                                    st.markdown(f'<span class="price-tag">₹{discounted:,.0f}</span> <span class="old-price">₹{retail:,.0f}</span>', unsafe_allow_html=True)
                                elif discounted is not None:
                                    st.markdown(f'<span class="price-tag">₹{discounted:,.0f}</span>', unsafe_allow_html=True)
                                
                                # View Button
                                website_url = result.get("website_url")
                                if website_url:
                                    st.link_button("View Product ↗", website_url, use_container_width=True)

prompt = st.chat_input("Ask me to find similar sarees (e.g. 'Find more red banarasi sarees like this')...")

if prompt:
    if not st.session_state.query_path:
        st.error("Please upload a saree image in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        agent.image_path = Path(st.session_state.query_path)
        with st.chat_message("assistant"):
            with st.spinner("Searching the catalogue..."):
                response = agent.chat(prompt)
            st.markdown(response["message"])

        # Attach the results array to the chat history dictionary
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response["message"],
            "results": response.get("results", [])
        })
        
        st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    """
    <div style="text-align:center; color: var(--text-color); opacity: 0.6; font-size: 0.8rem; padding: 1rem;">
        TailorTalk · FashionCLIP + FAISS · AI-powered visual search
    </div>
    """,
    unsafe_allow_html=True
)