import streamlit as st
import os
from glob import glob
import base64
import random

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def get_random_image_from_shape_type(angle_dir, shape_type_dir):
    """Lấy một hình ảnh ngẫu nhiên từ thư mục shape_type"""
    # Tạo đường dẫn đầy đủ đến thư mục shape_type
    full_path = os.path.join(angle_dir, shape_type_dir)
    # Lấy tất cả các file trong thư mục
    image_files = []
    if os.path.exists(full_path):
        # Lấy tất cả các file trong thư mục
        image_files = [f for f in os.listdir(full_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        # Tạo đường dẫn đầy đủ cho mỗi file
        image_files = [os.path.join(full_path, f) for f in image_files]
    
    return random.choice(image_files) if image_files else None

def main():
    st.set_page_config(page_title="SuperPoint Analysis", layout="wide")
    
    # Tiêu đề chính
    st.title("Phân tích và đánh giá SuperPoint")
    
    # 1. Phần giới thiệu dataset
    st.header("1. Giới thiệu Dataset")
    
    # Container cho 8 hình ảnh phía trên
    st.write("#### Hình ảnh minh họa các keypoints được phát hiện")
    
    # Tạo 7 cột cho mỗi hàng
    cols_top = st.columns(7)
    
    # Danh sách đường dẫn đến các hình ảnh của bạn
    image_paths = [
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_checkerboard_original.png",
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_cube_original.png",
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_polygon_original.png",
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_lines_original.png",
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_multiple_polygons_original.png",
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_star_original.png",
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_stripes_original.png",
        # Hàng 2
        "UIUX/FeaturesMatching/detector_results5/draw_checkerboard_result.png",
        "UIUX/FeaturesMatching/detector_results5/draw_cube_result.png",
        "UIUX/FeaturesMatching/detector_results5/draw_polygon_result.png",
        "UIUX/FeaturesMatching/detector_results5/draw_lines_result.png",
        "UIUX/FeaturesMatching/detector_results5/draw_multiple_polygons_result.png",
        "UIUX/FeaturesMatching/detector_results5/draw_star_result.png",
        "UIUX/FeaturesMatching/detector_results5/draw_stripes_result.png",
    ]
    
    # Hiển thị hàng đầu tiên (7 ảnh)
    for i, col in enumerate(cols_top):
        with col:
            st.image(
                image_paths[i],
                caption=f"Ảnh gốc {i + 1}",
                use_column_width=True
            )
    
    # Tạo 7 cột cho hàng thứ hai
    cols_bottom = st.columns(7)
    
    # Hiển thị hàng thứ hai (7 ảnh)
    for i, col in enumerate(cols_bottom):
        with col:
            st.image(
                image_paths[i + 7],  # Lấy 7 ảnh tiếp theo
                caption=f"Keypoints Groundtruth {i + 8}",
                use_column_width=True
            )
    
    # Nội dung giới thiệu dataset
    st.write("""
    ### Tổng quan về Synthetic Dataset
    
    Dataset tổng hợp được tạo ra để huấn luyện và đánh giá mô hình SuperPoint, bao gồm nhiều loại hình học cơ bản khác nhau:

    1. **Cấu trúc Dataset:**
    Dataset bao gồm các thư mục con sau: draw_checkerboard, draw_cube, draw_ellipses, draw_lines, draw_multiple_polygons, draw_polygon, draw_star, draw_stripes, gaussian_noise

    Mỗi thư mục con đều chứa:
    - Thư mục `images/`: Lưu trữ các ảnh synthetic
    - Thư mục `points/`: Chứa các file .npy tương ứng lưu tọa độ groundtruth keypoints của mỗi ảnh
    
    2. **Đặc điểm của Dataset:**
    - Mỗi thư mục con đại diện cho một loại hình học khác nhau (đường thẳng, đa giác, hình sao,...)
    - Tất cả ảnh được tạo tự động (synthetic) với các keypoints được xác định chính xác
    - Mỗi ảnh đều có một file .npy tương ứng chứa thông tin về vị trí các keypoints

    3. **Format dữ liệu:**
    - Ảnh được lưu dưới dạng grayscale với kích thước 240x320 pixels
    - File .npy chứa mảng numpy 2 chiều với shape (N, 2), trong đó N là số lượng keypoints
    - Mỗi keypoint được biểu diễn bởi tọa độ (x, y) trong ảnh
    """)
    

    # 2. Phần phương pháp
    st.header("2. Phương pháp")
    st.write("""
    ### Giới thiệu về SuperPoint
    SuperPoint là một mô hình deep learning được giới thiệu vào năm 2018 trong bài báo "SuperPoint: Self-Supervised Interest Point Detection and Description" bởi DeTone, Malisiewicz và Rabinovich tại Magic Leap, Inc. Bài báo được công bố tại hội nghị CVPR Workshop 2018.

    ### Đặc điểm nổi bật
    - Là một trong những mô hình đầu tiên áp dụng deep learning vào bài toán phát hiện và mô tả đặc trưng
    - Sử dụng phương pháp học tự giám sát (self-supervised learning)
    - Có khả năng hoạt động real-time
    """)

    # Hiển thị hình ảnh kiến trúc mạng
    st.write("### Kiến trúc tổng quan của SuperPoint")
    st.image("UIUX/FeaturesMatching/keke.jpg", caption="Kiến trúc mạng SuperPoint", use_column_width=True)

    st.write("""
    ### Kiến trúc mạng
    SuperPoint đợc thiết kế với kiến trúc đa nhiệm, bao gồm:

    1. **Shared Encoder:**
    - Sử dụng kiến trúc VGG-style
    - 8 lớp tích chập (convolutional layers)
    - Giảm độ phân giải ảnh xuống 1/8
    - Nhận đầu vào là ảnh grayscale kích thước H×W×1
    
    2. **Interest Point Decoder (Detector Head):**
    - Phát hiện các điểm đặc trưng (keypoints)
    - Tạo ra heatmap với kích thước H/8 × W/8 × 65
    - Sử dụng lớp Softmax để tính xác suất điểm đặc trưng
    - Reshape để tạo bản đồ điểm đặc trưng cuối cùng
    
    3. **Descriptor Decoder (Descriptor Head):**
    - Tạo vector đặc trưng cho mỗi keypoint
    - Kích thước descriptor: 256 chiều (D=256)
    - Sử dụng phép nội suy Bi-Cubic
    - Chuẩn hóa L2 để tạo ra các descriptor có độ dài cố định

    ### Quy trình xử lý
    1. Ảnh đầu vào được đưa qua Shared Encoder để trích xuất đặc trưng
    2. Đặc trưng được chia thành 2 nhánh xử lý song song:
       - Nhánh Interest Point Decoder: phát hiện vị trí các điểm đặc trưng
       - Nhánh Descriptor Decoder: tạo ra các vector mô tả cho mỗi điểm
    """)
    
    # Định nghĩa style cho container
    container_style = """
        <div style='display: flex; flex-direction: column; height: 500px; align-items: center;'>
            <h4 style='text-align: center; height: 50px; margin: 10px 0;'>
                {}
            </h4>
            <div style='flex: 1; display: flex; align-items: center; width: 100%; padding: 10px;'>
                <img src='{}' style='width: 100%; max-height: 350px; object-fit: contain;'>
            </div>
            <p style='text-align: center; height: 30px; margin: 10px 0;'>
                {}
            </p>
        </div>
    """
    # Hiển thị hình ảnh ở giữa
    st.image("UIUX/FeaturesMatching/OIP.jpg", 
             caption="Quá trình trích xuất đặc trưng", 
             use_column_width=True)
    
    # 3. Phần phương pháp đánh giá
    st.header("3. Phương pháp đánh giá")
    st.write("""
    ### Quy trình đánh giá
    Để đánh giá hiệu suất của SuperPoint và so sánh với các phương pháp truyền thống như SIFT và ORB trên tập dữ liệu Synthetic Shapes Dataset, chúng tôi thực hiện các bước sau:

    1. **Trích xuất đặc trưng:**
    - Trích xuất vector đặc trưng sử dụng SIFT, ORB và SuperPoint
    - Thực hiện trên các keypoint ground truth ở các góc quay khác nhau của ảnh
    
    2. **So khớp keypoint:**
    - Sử dụng phương pháp Brute-Force Matching để so khớp các vector đặc trưng
    - So sánh giữa ảnh gốc và các ảnh đã quay
    
    3. **Đánh giá kết quả:**
    - Tính toán phần trăm các keypoint được so khớp chính xác
    - So sánh hiệu suất giữa các phương pháp ở mỗi góc quay
    - Phân tích độ ổn định của các phương pháp dưới ảnh hưởng của phép quay
    """)

    # Hiển thị hình ảnh minh họa quy trình đánh giá
    st.image("UIUX/FeaturesMatching/34e1ed465aa5420d82438c0e6ad330a2.png", 
             caption="Quá trình so khớp các đặc trưng", 
             use_column_width=True)
    
    # 4. Phần kết quả thí nghiệm
    st.header("4. Kết quả thí nghiệm")
    st.write("### Kết quả trên tập kiểm thử")
    st.write("Kết quả matching của SuperPoint với các góc quay khác nhau")

    angle = st.slider("Chọn góc quay", 0, 60, 0, 10)

    # Xác định thư mục góc tương ứng
    base_dir = "UIUX/FeaturesMatching/superpoint_shape_type_images_20241130_094216"
    angle_dir = os.path.join(base_dir, f"angle_{angle}")

    # Tạo danh sách các thư mục shape_type
    shape_type_dirs = [f"shape_type_{i}" for i in range(8)]

    # Tạo layout 4x2 (2 hàng, mỗi hàng 4 ảnh)
    row1_cols = st.columns(3)
    row2_cols = st.columns(3)
    row3_cols = st.columns(2)  # Hàng cuối chỉ có 2 ảnh

    # Duyệt qua từng shape_type
    for idx, shape_type_dir in enumerate(shape_type_dirs):
        img_path = get_random_image_from_shape_type(angle_dir, shape_type_dir)
        
        if img_path:
            # Xác định hàng và cột cho mỗi ảnh
            if idx < 3:  # 3 ảnh đầu vào hàng 1
                col = row1_cols[idx]
            elif idx < 6:  # 3 ảnh tiếp vào hàng 2
                col = row2_cols[idx - 3]
            else:  # 2 ảnh cuối vào hàng 3
                col = row3_cols[idx - 6]
            
            with col:
                try:
                    st.image(img_path, 
                            caption=f"Ảnh kết quả {idx + 1}", 
                            use_column_width=True)
                except Exception as e:
                    st.error(f"Lỗi khi hiển thị ảnh: {str(e)}")

    # Thêm phần hiển thị biểu đồ đánh giá
    st.write("### Kết quả đánh giá")
    st.write("Biểu đồ so sánh độ chính xác (Accuracy) giữa các phương pháp theo góc quay")

    # Hiển thị biểu đồ
    st.image("UIUX/FeaturesMatching/matching.png", 
             caption="Biểu đồ Accuracy theo góc quay", 
             use_column_width=True)

    # Thêm phần giải thích biểu đồ
    st.write("""
    #### Nhận xét về kết quả thực nghiệm:

    1. **Hiệu suất của SuperPoint:**
       - Đạt độ chính xác cao nhất trong cả ba phương pháp ở hầu hết các góc quay
       - Duy trì độ ổn định tốt khi góc quay tăng từ 0° đến 40°
       - Đặc biệt hiệu quả trong khoảng góc 0-20°, với độ chính xác trên 80%
       - Khả năng chống chịu với biến đổi hình học tốt hơn so với các phương pháp truyền thống

    2. **So sánh với các phương pháp truyền thống:**
       - Tại góc 0°: Cả ba phương pháp đều đạt hiệu suất tối đa (100%), cho thấy độ tin cậy cao trong điều kiện không có biến đổi
       - Khi góc quay tăng:
         + SuperPoint giảm hiệu suất chậm hơn so với SIFT và ORB
         + SIFT có xu hướng giảm mạnh nhất
         + ORB duy trì được độ ổn định ở góc lớn nhưng độ chính xác thấp hơn SuperPoint

    3. **Ưu điểm của SuperPoint:**
       - Khả năng trích xuất đặc trưng bền vững với các biến đổi hình học
       - Độ chính xác cao và ổn định trong nhiều điều kiện khác nhau
       - Đặc biệt phù hợp cho các ứng dụng yêu cầu độ chính xác cao ở góc quay nhỏ và trung bình

    4. **Hạn chế cần cải thiện:**
       - Hiệu suất giảm đáng kể ở các góc quay lớn (trên 60°)
       - Cần cải thiện khả năng nhận dạng đặc trưng ở các góc quay cực đại
       - Thời gian xử lý có thể cải thiện thêm để tối ưu cho ứng dụng thời gian thực
    """)

if __name__ == "__main__":
    main()
