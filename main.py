import streamlit as st
import importlib
import os

def load_page(page_module):
    """Dynamically import and run page module"""
    try:
        module = importlib.import_module(page_module)
        if hasattr(module, 'main'):
            module.main()
    except Exception as e:
        st.error(f"Error loading page {page_module}: {str(e)}")

def main():
    # Cấu hình trang
    st.set_page_config(
        page_title="Computer Vision Applications",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Định nghĩa các trang
    PAGES = {
        "🏠 Home": "home",
        "1️⃣ Ứng dụng tách nền bằng thuật toán GrabCut": "pages.grabcut",
        "2️⃣ Phân đoạn ký tự bằng Watershed Segmentation": "pages.watershed",
        "3️⃣ Phát hiện khuôn mặt với Haar Features và KNN": "pages.haar_knn",
        "4️⃣ Ứng dụng xác nhận khuôn mặt": "pages.face_verification",
        "5️⃣ Phát hiện Keypoint trên Synthetic Shapes Dataset": "pages.keypoint_detection",
        "6️⃣ So khớp Keypoint dựa trên tiêu chí Rotation": "pages.keypoint_matching",
        "7️⃣ Tìm kiếm ảnh chứa đối tượng truy vấn": "pages.instance_search",
        "8️⃣ Theo dõi đối tượng (SOT) bằng thuật toán KCF": "pages.tracking",
        "9️⃣ Thuật toán SORT (Simple Online Realtime Tracking)": "pages.sort_mot"
    }

    # Sidebar navigation
    st.sidebar.title("Navigation 🧭")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Hiển thị trang được chọn
    if selection == "🏠 Home":
        # Hiển thị trang chủ
        st.title("🖼️ Computer Vision Applications")
        
        st.markdown("""
        ## Welcome to Our Computer Vision Suite!
        
        This application showcases various computer vision techniques and algorithms:
        
        ### Available Applications:
        
        1. **GrabCut Segmentation** 🎯
           - Interactive foreground extraction
           - User-guided segmentation
        
        2. **Watershed Segmentation** 💧
           - Automatic image segmentation
           - License plate character detection
        
        3. **Face Detection (Haar + KNN)** 👤
           - Real-time face detection
           - KNN classification
        
        4. **Face Verification** 🔍
           - Face recognition and verification
           - Deep learning based approach
        
        5. **Keypoint Detection** 🎯
           - Feature point detection
           - Synthetic shapes analysis
        
        6. **Keypoint Matching** 🔄
           - Rotation-based matching
           - Feature correspondence
        
        7. **Instance Search** 🔎
           - Object-based image retrieval
           - Feature matching and ranking
        
        8. **Single Object Tracking (KCF)** 🎯
           - Real-time object tracking
           - Kernelized Correlation Filters
        
        9. **Multiple Object Tracking (SORT)** 👥
           - Track multiple objects simultaneously
           - Simple Online and Realtime Tracking
        """)

        # Thêm phần Technologies Used
        st.markdown("""
        ### 🛠️ Technologies Used
        - **OpenCV**: Core computer vision operations
        - **Streamlit**: Interactive web interface
        - **NumPy**: Numerical computations
        - **Deep Learning**: Advanced detection and recognition
        """)

        # Footer
        st.markdown("""
        ---
        ### 📚 References & Documentation
        For detailed documentation and references, please visit the respective application pages.
        """)

    else:
        # Load trang được chọn
        page_module = PAGES[selection]
        load_page(page_module)

    # Thêm footer trong sidebar

if __name__ == "__main__":
    main()