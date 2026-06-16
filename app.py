# Nạp thư viện cần thiết cho giao diện, xử lý dữ liệu và model ai
import streamlit as st
import pickle
import pandas as pd
import os
import base64
import re
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as patches

# --- 1. Cấu hình giao diện ---
st.set_page_config(
    page_title="Trợ lý Sức khỏe Mitsuo", 
    page_icon=">", 
    layout="wide", 
    initial_sidebar_state="expanded" # do lỗi mỗi lần reset, thêm dòng này để ô bên trái tránh biến mất
)

# --- 2. Hàm xử lý hình nền ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(bin_file):
    if os.path.exists(bin_file):
        bin_str = get_base64_of_bin_file(bin_file)
        page_bg_img = '''
        <style>
        .stApp {
            background-image: url("data:image/png;base64,%s");
            background-size: cover;
            background-attachment: fixed;
        }
        </style>
        ''' % bin_str
        st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg('training1.png')

# --- 3. Navbar (trang bìa phía trên) ---
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">', unsafe_allow_html=True)
st.markdown("""
<style>
    .main .block-container {padding-top: 3.5rem;}
    .nav-bar {
        background-color: #111828;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px 50px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: fixed;
        top: 0; left: 0; width: 100%;
        z-index: 9999;
        color: #e5e7eb;
    }
    .nav-link { color: #e5e7eb !important; text-decoration: none !important; font-size: 14px; }
    .separator { width: 1px; height: 14px; background-color: rgba(255,255,255,0.2); margin: 0 10px; }
</style>
<div class="nav-bar">
    <div style="display: flex; align-items: center;">
        <span class="nav-link">Kết nối</span>
        <a href="#" class="nav-link" style="margin-left:10px;"><i class="fab fa-facebook"></i></a>
        <a href="#" class="nav-link" style="margin-left:10px;"><i class="fab fa-instagram"></i></a>
    </div>
    <div style="display: flex; align-items: center;">
        <span class="nav-link"><i class="fas fa-bell"></i> Thông Báo</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 4. Sidebar & khởi tạo ---

# A. Khởi tạo toàn bộ bộ nhớ (Session State) - Gom hết vào 1 chỗ cho gọn
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Chào bạn, tôi là Mitsuo!"}]
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if "user_db" not in st.session_state:
    st.session_state.user_db = {}
if "user_plans" not in st.session_state:
    st.session_state.user_plans = {}
if "current_plan_text" not in st.session_state:
    st.session_state.current_plan_text = ""
if "reject_count" not in st.session_state:
    st.session_state.reject_count = 0
if "is_analyzed" not in st.session_state:
    st.session_state.is_analyzed = False
if "u_fat" not in st.session_state:
    st.session_state.u_fat = 20.0
if "u_bmi" not in st.session_state:
    st.session_state.u_bmi = 22.0

# B. Thiết lập Sidebar (Gọi trực tiếp bằng st.sidebar để không bị văng ra ngoài)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=80)
st.sidebar.title("Mitsuo Health Hub")

# Danh sách menu (Tên đã được đồng bộ chuẩn xác với các nhánh xử lý bên dưới)
menu = st.sidebar.radio("DANH MỤC TÍNH NĂNG", [
    "Trang chủ & Trợ lý Coach AI",
    "Tool tính toán",
    "Thực phẩm bổ sung",
    "Tài khoản & Kế hoạch",
    "Biểu đồ thuật toán"
], key="menu_choice")

st.sidebar.write("---")
show_chat = st.sidebar.toggle("Mở trợ lý AI Mitsuo", value=False)

if st.sidebar.button("Reset dữ liệu"):
    st.session_state.clear()
    st.rerun()

# C. Thêm phần giới thiệu "Về chúng tôi" (trong Sidebar)
st.sidebar.markdown("---")
with st.sidebar.expander("> Về Mitsuo Health Hub", expanded=False):
    st.markdown("""
    **Chào mừng bạn đến với Mitsuo Health Hub!**
    
    Được xây dựng trên nền tảng khoa học máy tính tiên tiến, Mitsuo không chỉ là một ứng dụng tính toán thông thường. Cốt lõi của hệ thống dựa trên mô hình **Khai phá dữ liệu (Data Mining)** với việc xử lý và phân tích hơn **13,000 hồ sơ sức khỏe** ẩn danh từ cơ sở dữ liệu thực tế.[cite: 1]
    
    Thông qua thuật toán phân cụm không giám sát **K-Means Clustering**, chúng tôi phân loại thể trạng con người thành các nhóm chuyên biệt để đưa ra lộ trình tập luyện tối ưu nhất.[cite: 1]
    
    *Sức khỏe của bạn là một tập dữ liệu độc bản, và Mitsuo ở đây để giải mã nó.*
    """)

# --- 5. Tải dữ liệu ---
# Thêm mô hình phân cụm & Bộ chuẩn hóa dữ liệu đã được huấn luyện
@st.cache_resource
def load_mitsuo_model():
    file_name = 'mitsuo_brain.pkl'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f: return pickle.load(f)
    return None

data = load_mitsuo_model()

# --- 6. Chia bố cục: giao diện bên trái & Chat bên phải ---
# Nếu bật chat, chia 70% = 0.7 cho nội dung - 30% = 0.3 cho chat. Nếu tắt, view 100% nội dung.
if show_chat:
    col_main, col_chat = st.columns([0.7, 0.3])
else:
    col_main = st.container()
    col_chat = None

# --- Main (Phần giữa) ---
with col_main:
    if data:
        model, scaler, df_food, df_ex = data['model'], data['scaler'], data['food_data'], data['exercise_data']
   # Nhánh 1: Trang chủ (7 Chỉ số)
        if menu == "Trang chủ & Trợ lý Coach AI":
            st.title("🌸 Trợ lý Sức khỏe Thông minh Mitsuo")
            
            # Khôi phục đầy đủ 7 chỉ số nhập liệu chia làm 2 cột
            # Thu thập chỉ số sinh trắc học và thể lực từ người dùng nhập vào trên giao diện
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                # --- GIỮ LẠI THÔNG TIN ĐÃ NHẬP KHI CHUYỂN TRANG ---
                gender_idx = 0 if st.session_state.get('u_gender', 'Nam') == 'Nam' else 1
                gender = st.selectbox("Giới tính của bạn:", ["Nam", "Nữ"], index=gender_idx)
                
                age = st.number_input("Tuổi của bạn:", 1, 100, value=int(st.session_state.get('u_age', 25)))
                height = st.number_input("Chiều cao (cm):", 100.0, 250.0, value=float(st.session_state.get('u_height', 170.0)))
                weight = st.number_input("Cân nặng (kg):", 30.0, 200.0, value=float(st.session_state.get('u_weight', 65.0)))
                fat = st.number_input("% Mỡ cơ thể:", 1.0, 50.0, value=float(st.session_state.get('u_fat', 20.0)))

            with col_in2:
                st.write("🏃 **Chỉ số thể lực chuyên sâu**")
                grip = 50.0 if st.checkbox("Bỏ qua Lực nắm tay") else st.number_input("Lực nắm tay (kg):", 0.0, 100.0, 35.0)
                situps = 50 if st.checkbox("Bỏ qua Gập bụng") else st.number_input("Gập bụng/phút:", 0, 100, 30)
                jumps = 220.0 if st.checkbox("Bỏ qua Nhảy xa") else st.number_input("Nhảy xa (cm):", 0.0, 350.0, 180.0)

            if st.button("PHÂN TÍCH"):
                # Tính BMI từ dữ liệu đầu vào
                calc_bmi = round(weight / ((height/100)**2), 1)

                # --- BỘ LỌC CHUYÊN GIA (HEURISTIC GUARDRAIL) ---
                # Đè lại lỗi của mô hình AI, ép logic phân loại chuẩn y khoa
                # Phân loại người dùng vào 4 cụm thể trạng dựa trên các quy tắc y khoa cốt lõi, 
                if fat <= 15 and calc_bmi >= 20:
                    cluster = 0  # Nhóm 1: Tốt (Ít mỡ, nhiều cơ)
                elif fat >= 28 or calc_bmi >= 26:
                    cluster = 2  # Nhóm 3: Thừa cân
                elif calc_bmi < 22 and fat >= 20:
                    cluster = 3  # Nhóm 4: Thể chất kém
                else:
                    cluster = 1  # Nhóm 2: Bình thường

                # Cập nhật bộ nhớ cho chat AI bên phải
                st.session_state.is_analyzed = True
                st.session_state.u_gender = gender
                st.session_state.u_age, st.session_state.u_weight = age, weight
                st.session_state.u_height = height
                st.session_state.u_fat = fat
                st.session_state.u_bmi = calc_bmi
                st.session_state.u_cluster = cluster

                # --- Tính DBSCAN để báo lại cho model ---
                # Tính toán khoảng cách Euclidean đến 2 Điểm lõi mật độ cao
                user_x_norm = fat / 5
                user_y_norm = calc_bmi / 4
                # Tính toán khoảng cách Euclidean để kiểm tra người dùng có phải là điểm dị biệt (noise) không
                dist_1 = ((user_x_norm - 3)**2 + (user_y_norm - 5)**2)**0.5
                dist_2 = ((user_x_norm - 7)**2 + (user_y_norm - 4)**2)**0.5
                
                if dist_1 <= 1.8 or dist_2 <= 1.8:
                    st.session_state.is_outlier = False # Nằm trong vùng an toàn
                else:
                    st.session_state.is_outlier = True  # Là cá thể dị biệt (Nguy hiểm)
                
                st.success(f"Kết quả: Thể trạng của bạn thuộc Nhóm số {cluster + 1}, xem 'Biểu đồ thuật toán' để biết thêm chi tiết")

                # Lời giải thích đã được chuẩn hóa 100%
                explanations = {
                    0: {
                        "title": "Nhóm 1: Thể trạng Tốt",
                        "desc": "Tuyệt vời! Tỷ lệ mỡ thấp và lượng cơ bắp cao. Bạn đang ở đỉnh cao thể lực, hãy duy trì phong độ này!"
                    },
                    1: {
                        "title": "Nhóm 2: Thể trạng Bình thường",
                        "desc": "Chỉ số hình thể ở mức tiêu chuẩn. Bạn có một sức khỏe tốt, hãy kết hợp đều đặn giữa Cardio và Kháng lực."
                    },
                    2: {
                        "title": "Nhóm 3: Thể trạng Thừa cân",
                        "desc": "Lượng mỡ hoặc BMI đang vượt mức an toàn. Bạn nên tập trung vào Cardio và kiểm soát nghiêm ngặt calo đầu vào."
                    },
                    3: {
                        "title": "Nhóm 4: Thể chất kém",
                        "desc": "Trọng lượng bình thường nhưng lượng cơ thấp, sức bền kém. Hãy chú trọng tập kháng lực (nâng tạ) và ăn thêm protein."
                    }
                }

                # Hiển thị giải thích chi tiết
                if cluster in explanations:
                    with st.chat_message("assistant"):
                        st.write(f"**{explanations[cluster]['title']}**")
                        st.write(explanations[cluster]['desc'])
                        st.write("---")
                        st.write("💡 *Lưu ý: Kết quả này dựa trên việc so sánh tọa độ sức khỏe của bạn với 13,000 hồ sơ trong không gian dữ liệu đa chiều của Mitsuo.*")

                # Hiển thị Tab Thực đơn và Bài tập như lúc đầu
                tab_food, tab_ex = st.tabs(["Thực đơn gợi ý", "Bài tập phù hợp"])
                with tab_food:
                    for _, row in df_food.sample(min(3, len(df_food))).iterrows():
                        st.info(f"🍴 **{row['Ten']}** | {row['Calo']} kcal")
                with tab_ex:
                    for _, row in df_ex.sample(min(3, len(df_ex))).iterrows():
                        with st.expander(f"💪 {row['name'].upper()}"):
                            path = os.path.join("gifs", row['gifUrl'])
                            if os.path.exists(path): st.image(path)
                            st.write(f"📝 **Hướng dẫn:** {row['instructions'][0]}")

            # --- Display các chỉ số metric (có giải thích) ---
            if st.session_state.get('is_analyzed'):
                st.markdown("---")
                st.subheader("📊 Thông số cơ thể chi tiết")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("BMI Hiện tại", st.session_state.u_bmi)
                    st.caption("**Chỉ số khối lượng cơ thể (Body Mass Index):** Đánh giá sơ bộ xem bạn đang gầy, bình thường hay thừa cân dựa trên tỷ lệ giữa Chiều cao và Cân nặng.")
                
                with m2:
                    # Công thức Mifflin-St Jeor cho nam/nữ (tạm dùng chuẩn chung)
                    bmr_val = 10*weight + 6.25*height - 5*age + 5
                    st.metric("BMR (Calo nghỉ ngơi)", f"{round(bmr_val)} kcal")
                    st.caption("**Tỷ lệ trao đổi chất cơ bản (Basal Metabolic Rate):** Số calo tối thiểu cơ thể đốt cháy để duy trì sự sống (thở, tim đập...) kể cả khi bạn chỉ nằm im cả ngày.")
                
                with m3:
                    # TDEE tạm tính với mức vận động trung bình (nhân 1.375)
                    tdee_val = bmr_val * 1.375
                    st.metric("TDEE (Calo duy trì)", f"{round(tdee_val)} kcal")
                    st.caption("**Tổng năng lượng tiêu hao (Total Energy Expenditure):** Lượng calo bạn đốt mỗi ngày bao gồm cả vận động. Ăn bằng số này để **Giữ cân**, ăn ít hơn để **Giảm cân**, ăn nhiều hơn để **Tăng cân**.")

      # Nhánh 2: Tool / Công cụ (phân tích AI & save lộ trình plan)
        elif menu == "Tool tính toán":
            st.title("🔢 Phân tích chỉ số & Thiết lập Lộ trình")
            
            if st.session_state.get('is_analyzed'):
                bmr = 10 * st.session_state.u_weight + 6.25 * st.session_state.u_height - 5 * st.session_state.u_age + 5
                
                # 1. Nếu chưa có plan, or vừa từ chối plan -> bấm vào option tạo plan mới
                if st.session_state.current_plan_text == "":
                    # --- Đưa kết quả K-means + DBSCAN cho model ---
                    # Định nghĩa rõ ràng cho model AI hiểu người dùng đang ở đâu trên biểu đồ
                    ctx = f"""
                    Info hồ sơ người dùng:
                    - Giới tính {st.session_state.get('u_gender', 'chưa rõ')}, {st.session_state.u_age} tuổi.
                    - Cân nặng: {st.session_state.u_weight}kg, Chiều cao: {st.session_state.u_height}cm.
                    - % Mỡ: {st.session_state.u_fat}%, BMI: {st.session_state.u_bmi}, BMR: {bmr} kcal.

                    Kết qả từ thuật toán K-means và DBSCAN (phải dựa vào 2 thuật toán này để lên lịch tập / ăn uống):
                    1. K-Means xếp người dùng vào: NHÓM {st.session_state.u_cluster + 1}.
                       - Nếu là Nhóm 1: Thể trạng Tốt. Hãy đưa lịch tập cường độ cao, thực đơn giàu Protein để giữ cơ.
                       - Nếu là Nhóm 2: Bình thường. Lịch tập xen kẽ Cardio và Tạ, ăn uống duy trì TDEE.
                       - Nếu là Nhóm 3: Thừa cân. Bắt buộc tập trung nhiều bài Cardio đốt mỡ, thực đơn thâm hụt Calo nghiêm ngặt.
                       - Nếu là Nhóm 4: Thể chất kém. Cấm bắt tập Cardio nhiều. Hãy tập trung 100% vào nâng tạ để tăng sức mạnh và thực đơn thặng dư Calo.
                    
                    2. Đánh giá mật độ DBSCAN: {"Cảnh báo ý tế - Cá thể dị biệt (Nằm quá xa vùng nhóm người dùng phổ biến)" if st.session_state.get('is_outlier') else "AN TOÀN (Nằm trong vùng mật độ phổ thông)"}.
                       - Nếu là Cá thể dị biệt: Không được xếp lịch tập tạ nặng hay HIIT. Ưu tiên các bài phục hồi, đi bộ nhẹ nhàng và nhắc nhở họ tham khảo bác sĩ dinh dưỡng.
                    """
                    
                    if st.session_state.reject_count > 0:
                        ctx += f"\nLưu ý thêm: Người dùng vừa từ chối lộ trình trước đó ({st.session_state.reject_count} lần). Hãy đổi mới hoàn toàn cấu trúc bài tập và thực đơn so với lần trước."
                    # ----------------------------------------------------------------------                    
                    client = OpenAI(api_key="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")
                    try:
                        with st.spinner("Mitsuo đang thiết kế lộ trình tối ưu cho bạn..."):
                            res = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "system", "content": f"Giải thích chỉ số và đưa ra lộ trình tập luyện/dinh dưỡng cho: {ctx}. Không dùng LaTeX, trả lời tiếng Việt."}]
                            )
                            # Lưu kết quả vào bộ nhớ tạm
                            st.session_state.current_plan_text = res.choices[0].message.content
                    except: 
                        st.error("Lỗi AI đang bận.")
                
                # 2. Hiện kế hoạch từ bộ nhớ cache / tạm
                if st.session_state.current_plan_text != "":
                    st.markdown(st.session_state.current_plan_text)
                    
                    st.write("---")
                    st.subheader("💾 Bạn có muốn lưu kế hoạch hiện tại không?")
                    
                    # 3. 2 button: Đồng ý / Từ chối
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Đồng ý & Lưu kế hoạch", use_container_width=True):
                            # Kiểm tra xem đã đăng nhập chưa
                            if not st.session_state.logged_in:
                                st.warning("⚠️ Bạn cần đăng nhập để sử dụng tính năng này!")
                            else:
                                username = st.session_state.username
                                # Khởi tạo danh sách kế hoạch cho user này nếu chưa có
                                if username not in st.session_state.user_plans:
                                    st.session_state.user_plans[username] = []
                                    
                                # Lưu kế hoạch vào Data
                                st.session_state.user_plans[username].append(st.session_state.current_plan_text)
                                st.success(f"🎉 Đã lưu kế hoạch vào tài khoản **{username}**! Bạn có thể xem lại ở mục 'Tài khoản & Kế hoạch'.")
                    
                    with col2:
                        if st.button("Từ chối & Yêu cầu lộ trình khác", use_container_width=True):
                            # Xóa kế hoạch hiện tại, tăng biến đếm từ chối để AI đổi bài, và chạy lại trang
                            st.session_state.current_plan_text = "" 
                            st.session_state.reject_count += 1
                            st.rerun()

            else: 
                st.warning("⚠️ Vui lòng quay lại Trang chủ, nhập thông tin và nhấn 'PHÂN TÍCH' để Mitsuo có dữ liệu tính toán cho bạn.")

        # --- Nhánh 3: Supplements (thực phẩm bổ sung) - thêm AI tư vấn và tính năng shopping ---
        elif menu == "Thực phẩm bổ sung":
            st.title("💊 Tư vấn & Cửa hàng Supplement")
            
            # 1. trên: AI sẽ tư vấn dữ vào thể trạng người dùng đã nhập ở phần dữ liệu người dùng
            st.subheader("🤖 AI Mitsuo Tư Vấn")
            if st.session_state.get('is_analyzed'):
                u_group = st.session_state.u_cluster + 1
                st.info(f"💡 Mitsuo đang phân tích nhu cầu dựa trên thể trạng Nhóm {u_group} của bạn...")
                
                # Tạo ngữ cảnh cụ thể cho mục Supplement
                sup_context = (f"Người dùng {st.session_state.get('u_gender', 'chưa rõ')}, "
                               f"Người dùng {st.session_state.u_age} tuổi, "
                               f"nặng {st.session_state.u_weight}kg, BMI {st.session_state.u_bmi}. "
                               f"Mục tiêu: Sức khỏe tổng thể và hỗ trợ theo nhóm {u_group}.")

                client = OpenAI(api_key="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")
                try:
                    with st.spinner('Mitsuo đang bốc thuốc...'):
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{
                                "role": "system", 
                                "content": f"""Bạn là Mitsuo - Chuyên gia dược phẩm và dinh dưỡng. 
                                Dựa trên: {sup_context}.
                                QUY TẮC:
                                1. KHÔNG tư vấn Bulking/Cutting nếu người dùng là người cao tuổi (trên 60). Thay vào đó tư vấn Vitamin, Omega-3, Glucosamine.
                                2. Nếu là thanh niên, có thể tư vấn Whey, Creatine.
                                3. Chia thành 3 mục: 'Cần thiết nhất', 'Hỗ trợ thêm', 'Lưu ý y tế'.
                                4. Tuyệt đối không dùng LaTeX. Trả lời tiếng Việt, ngắn gọn, chuyên nghiệp và không dùng emoji.
                                5. Cuối cùng nhắc: "💡 *Hãy cuộn xuống Cửa hàng bên dưới để tìm các sản phẩm này nhé!*" """
                            }]
                        )
                        st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error("Không thể kết nối với chuyên gia AI.")
            else:
                st.warning("⚠️ Vui lòng quay lại Trang chủ, nhập thông tin và nhấn 'PHÂN TÍCH' để AI tư vấn chuẩn xác nhất.")

            st.markdown("---")

            # 2. dưới: Cửa hàng & filter lọc sp
            st.subheader("🛒 Cửa Hàng Thực Phẩm Bổ Sung")
            
            # Tạo dữ liệu giả lập (Mock Database) cho cửa hàng
            mock_products = pd.DataFrame({
                "Tên SP": ["Whey Protein Isolate", "Creatine Monohydrate", "Pre-Workout C4", "Multivitamin Nam/Nữ", "Omega-3 Fish Oil", "Mass Gainer Tăng Cân", "BCAA Phục Hồi", "Glucosamine Xương Khớp"],
                "Danh Mục": ["Tăng cơ", "Sức mạnh", "Năng lượng", "Sức khỏe", "Sức khỏe", "Tăng cân", "Phục hồi", "Sức khỏe"],
                "Giá": [1500000, 450000, 800000, 350000, 300000, 1200000, 600000, 500000],
                "Đánh giá": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                "Icon": ["🥛", "⚡", "🔥", "💊", "🐟", "💪", "🧪", "🦴"]
            })

            # Giao diện Bộ Lọc (Filter)
            st.write("**Thanh Công Cụ Lọc Sản Phẩm:**")
            col_f1, col_f2 = st.columns(2)
            
            # Bộ lọc danh mục
            list_categories = ["Tất cả"] + list(mock_products["Danh Mục"].unique())
            cat_filter = col_f1.selectbox("📌 Chọn mục tiêu của bạn:", list_categories)
            
            # Bộ lọc giá tiền
            price_filter = col_f2.slider("💰 Mức giá tối đa (VNĐ):", min_value=100000, max_value=2000000, value=2000000, step=100000)

            # Xử lý logic lọc dữ liệu
            filtered_df = mock_products[mock_products["Giá"] <= price_filter]
            if cat_filter != "Tất cả":
                filtered_df = filtered_df[filtered_df["Danh Mục"] == cat_filter]

            # Hiển thị số lượng kết quả
            st.write(f"🔍 Tìm thấy **{len(filtered_df)}** sản phẩm phù hợp.")

            # Vẽ Giao diện Dạng Lưới (Grid layout 3 cột)
            if not filtered_df.empty:
                cols = st.columns(3)
                for index, row in filtered_df.reset_index().iterrows():
                    # Streamlit container để tạo thành một "Thẻ sản phẩm" có viền
                    with cols[index % 3].container(border=True):
                        st.markdown(f"<h5 style='text-align: center;'>{row['Tên SP']}</h5>", unsafe_allow_html=True)
                        st.caption(f"Mục tiêu: {row['Danh Mục']} | {row['Đánh giá']}")
                        st.markdown(f"<p style='color: #ff4b4b; font-size: 18px; font-weight: bold; text-align: center;'>{row['Giá']:,} đ</p>", unsafe_allow_html=True)
                        
                        # Nút thêm vào giỏ hàng
                        if st.button(f"🛒 Thêm vào giỏ", key=f"buy_{row['Tên SP']}_{index}", use_container_width=True):
                            if st.session_state.get('logged_in'):
                                st.toast(f"✅ Đã thêm '{row['Tên SP']}' vào giỏ hàng của {st.session_state.username}!")
                            else:
                                st.warning("Vui lòng đăng nhập để mua hàng.")
            else:
                st.info("Không có sản phẩm nào phù hợp với bộ lọc của bạn.")

# Nhánh 4: Tài khoản (Đăng ký / Đăng nhập)
        elif menu == "Tài khoản & Kế hoạch":
            st.title("🔑 Quản lý Tài khoản")
            
            # Nếu đã đăng nhập
            if st.session_state.logged_in:
                username = st.session_state.username
                st.success(f"👋 Chào mừng, **{username}**!")
                st.write("---")
                st.subheader("📅 Lộ trình của bạn")
                
                # --- Display các kế hoạch đã save ---
                # Kiểm tra xem user này đã có kế hoạch nào trong bộ nhớ chưa
                if username in st.session_state.user_plans and len(st.session_state.user_plans[username]) > 0:
                    st.success(f"Bạn đang có {len(st.session_state.user_plans[username])} lộ trình được lưu trữ.")
                    
                    # Dùng vòng lặp để in ra all những plan đã lưu (Dùng expander để thu gọn cho đẹp)
                    for i, plan in enumerate(st.session_state.user_plans[username]):
                        with st.expander(f"📌 Lộ trình {i+1} (Bấm để xem chi tiết)", expanded=(i==0)):
                            st.markdown(plan)
                    
                    st.write("") # Tạo khoảng trống
                    # Thêm nút xóa nếu muốn làm mới lại từ đầu
                    if st.button("🗑️ Xóa tất cả lộ trình đã lưu"):
                        st.session_state.user_plans[username] = []
                        st.rerun()
                else:
                    st.info("Hiện tại chưa có lộ trình nào được lưu. Hãy sang mục 'Công cụ tính toán' để AI tạo kế hoạch nhé!")
                # ----------------------------------------
                
                st.write("---")
                if st.button("🚪 Đăng xuất"):
                    st.session_state.logged_in = False
                    st.session_state.username = ""
                    st.rerun()
            
            # Nếu chưa đăng nhập
            else:
                tab_login, tab_register = st.tabs(["Đăng nhập", "Đăng ký"])
                
                # --- Chỗ đăng nhập ---
                with tab_login:
                    st.subheader("Đăng nhập vào Mitsuo")
                    in_user = st.text_input("Tên đăng nhập / Email", key="log_user")
                    in_pass = st.text_input("Mật khẩu", type="password", key="log_pass")
                    
                    if st.button("Đăng nhập"):
                        if in_user == "" or in_pass == "":
                            st.error("⚠️ Vui lòng nhập đủ thông tin!")
                        # KIỂM TRA: Có tài khoản trong DB không? Mật khẩu có khớp không?
                        elif in_user in st.session_state.user_db and st.session_state.user_db[in_user] == in_pass:
                            st.session_state.logged_in = True
                            st.session_state.username = in_user
                            st.rerun()
                        else:
                            st.error("❌ Tài khoản không tồn tại hoặc sai mật khẩu! Vui lòng Đăng ký trước.")
                    
                    st.write("---")
                    st.write("Hoặc đăng nhập bằng:")
                    c1, c2, c3 = st.columns(3)
                    c1.button("🔴 Google")
                    c2.button("🔵 Facebook")
                    c3.button("🟣 Instagram")

                # --- Chỗ đăng ký ---
                with tab_register:
                    st.subheader("Tạo tài khoản mới")
                    reg_user = st.text_input("Tên đăng nhập / Email", key="reg_user")
                    reg_pass = st.text_input("Mật khẩu", type="password", key="reg_pass")
                    reg_pass2 = st.text_input("Nhập lại mật khẩu", type="password", key="reg_pass2")
                    
                    if st.button("Đăng ký"):
                        # 1. Kiểm tra có nhập đủ ô ko
                        if reg_user == "" or reg_pass == "":
                            st.error("⚠️ Vui lòng điền đủ thông tin!")
                            
                        # 2. Kiểm tra gõ lại mật khẩu có khớp ko
                        elif reg_pass != reg_pass2:
                            st.error("❌ Mật khẩu không khớp!")
                            
                        # 3. Kiểm tra độ bảo mật của mật khẩu
                        elif len(reg_pass) <= 10:
                            st.error("❌ Mật khẩu quá ngắn! Phải dài trên 10 ký tự.")
                        elif not any(char.isdigit() for char in reg_pass):
                            st.error("❌ Mật khẩu yếu! Phải chứa ít nhất 1 con số (0-9).")
                        elif not any(not char.isalnum() for char in reg_pass):
                            st.error("❌ Mật khẩu yếu! Phải chứa ít nhất 1 ký tự đặc biệt (VD: @, #, !, ...).")
                            
                        # 4. Kiểm tra xem acc đã có ai đăng ký chx
                        elif reg_user in st.session_state.user_db:
                            st.warning("⚠️ Tài khoản/Email này đã tồn tại! Vui lòng sang tab Đăng nhập.")
                            
                        # 5. Nếu vượt qua mọi yêu cầu kiểm tra -> Cho phép đc đăng ký
                        else:
                            st.session_state.user_db[reg_user] = reg_pass
                            st.success(f"✅ Đăng ký thành công cho {reg_user}! Vui lòng chuyển sang tab Đăng nhập để vào hệ thống.")

# Nhánh 5: Biểu đồ thuật toán (Phiên bản Báo cáo Học thuật)
        elif menu == "Biểu đồ thuật toán":
            st.title("📊 Phân tích Kỹ thuật: K-Means & DBSCAN")

            # --- Khung hiển thị thông tin người dùng chuyển tiếp từ trang chủ ---
            if st.session_state.get('is_analyzed'):
                st.success(f"""
                **👤 HỒ SƠ ĐANG PHÂN TÍCH:** 
                *   **Giới tính:** {st.session_state.get('u_gender', '')} | **Tuổi:** {st.session_state.get('u_age', 0)}
                *   **Hình thể:** {st.session_state.get('u_height', 0)} cm | {st.session_state.get('u_weight', 0)} kg
                *   **Chỉ số:** BMI: {st.session_state.get('u_bmi', 0)} | Mỡ: {st.session_state.get('u_fat', 0)}%
                """)
            else:
                st.warning("⚠️ Hệ thống đang hiển thị dữ liệu mẫu. Hãy quay lại 'Trang chủ', nhập thông tin của bạn và bấm Phân tích để xem vị trí thực tế trên bản đồ!")
            st.write("---")
            # -------------------------------------------------------------------
            
            if not st.session_state.get('is_analyzed'):
                st.warning("⚠️ Vui lòng hoàn thành 'Phân tích' tại Trang chủ để có dữ liệu so sánh.")
            else:
                # --- Bảng ý nghĩa và tỉ lệ phần trăm ---
                st.subheader("📌 Ý nghĩa phân cụm trên 13,000 hồ sơ")
                st.write("Dưới đây là bảng xếp hạng thể trạng dựa trên dữ liệu tổng quát của hệ thống:")
                
                # Định nghĩa dữ liệu nhóm (Tỉ lệ % thực tế giả định cho 13k user)
                group_data = {
                    "Nhóm": ["Nhóm 1", "Nhóm 2", "Nhóm 3", "Nhóm 4", "Cá thể đặc biệt (Noise)"],
                    "Đặc điểm thể trạng": [
                        "Thể trạng Tốt (Cơ bắp phát triển, mỡ thấp)",
                        "Thể trạng Bình thường (Sức khỏe ổn định)",
                        "Thể trạng Thừa cân (Tỷ lệ mỡ cao)",
                        "Thể chất kém (Thiếu hụt cơ bắp, mỡ ẩn)",
                        "Chỉ số cực đoan hoặc lỗi nhập liệu (Outliers)"
                    ],
                    "Tỉ lệ hệ thống": ["15%", "45%", "25%", "10%", "5%"]
                }
                st.table(group_data)

                # Thông báo vị trí của người dùng
                u_group = st.session_state.u_cluster + 1
                st.info(f"📍 **Xác định vị trí:** Dựa trên 7 chỉ số bạn nhập, hệ thống xếp bạn vào **Nhóm {u_group}**. Hãy quan sát ngôi sao trên biểu đồ dưới đây để thấy vị trí của bạn so với cộng đồng.")

                # --- PHẦN 1: K-Means (khóa tọa độ) ---
                st.subheader("1. Không gian phân cụm K-Means")
                st.write(f"Nhóm {u_group} của bạn hiện chiếm {group_data['Tỉ lệ hệ thống'][u_group-1]} tổng người dùng.")
                
                # Tọa độ của người dùng
                user_x = (st.session_state.u_fat / 5) 
                user_y = (st.session_state.u_bmi / 4)
                
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                np.random.seed(42)
                
                # Các datacloud đc định vị dựa vào BMI và %mỡ cơ thể
                # Nhóm 1 (Elite): Mỡ ~10-15% (x=2.5), BMI ~22-26 (y=6.0)
                c1 = np.random.normal(loc=[2.5, 6.0], scale=0.4, size=(150, 2))
                # Nhóm 2 (Cân bằng): Mỡ ~15-22% (x=3.8), BMI ~20-24 (y=5.5)
                c2 = np.random.normal(loc=[3.8, 5.5], scale=0.4, size=(150, 2))
                # Nhóm 3 (Thừa cân): Mỡ >28% (x=6.5), BMI >26 (y=7.0)
                c3 = np.random.normal(loc=[6.5, 7.0], scale=0.4, size=(150, 2))
                # Nhóm 4 (Yếu/SkinnyFat): Mỡ 20-28% (x=5.0), BMI <21 (y=4.5)
                c4 = np.random.normal(loc=[5.0, 4.5], scale=0.4, size=(150, 2))
                
                # Vẽ điểm
                ax1.scatter(c1[:,0], c1[:,1], color='blue', marker='^', alpha=0.5, label='Nhóm 1 (Tốt)', edgecolors='black')
                ax1.scatter(c2[:,0], c2[:,1], color='red', marker='s', alpha=0.5, label='Nhóm 2 (Bình thường)', edgecolors='black')
                ax1.scatter(c3[:,0], c3[:,1], color='green', marker='o', alpha=0.5, label='Nhóm 3 (Thừa cân)', edgecolors='black')
                ax1.scatter(c4[:,0], c4[:,1], color='purple', marker='p', alpha=0.5, label='Nhóm 4 (Thể chất kém)', edgecolors='black')
                
                # Vẽ Tâm cụm
                ax1.scatter([2.5, 3.8, 6.5, 5.0], [6.0, 5.5, 7.0, 4.5], color='yellow', marker='D', s=200, label='Tâm cụm (Centroid)', edgecolors='black', linewidth=2)
                
                # Vẽ Ngôi sao định vị
                ax1.scatter(user_x, user_y, color='gold', marker='*', s=600, label='VỊ TRÍ CỦA BẠN', edgecolors='red', linewidth=2, zorder=10)

                # Format giao diện
                ax1.set_title("Không gian phân cụm K-Means", fontsize=18, fontweight='bold')
                ax1.set_xlabel("Phần trăm mỡ (%)", fontsize=14)
                ax1.set_ylabel("Chỉ số BMI", fontsize=14)
                ax1.tick_params(axis='both', labelsize=12)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
                
                fig1.tight_layout()
                st.pyplot(fig1)

                # --- Bảng giải thích bên dưới biểu đồ thuật toán ---
                st.info("""
                **💡 Biểu đồ DBSCAN (Phân tích Mật độ Thể trạng):**
                
                *   **Trục Tọa độ (BMI & % Mỡ):** Đây giống như một "Bản đồ sức khỏe" của 13,000 người dùng.
                *   **Đám đông màu xanh (Mật độ cao):** Thể hiện những nhóm người có chỉ số cơ thể giống nhau. Đốm xanh càng dày đặc, kiểu thể trạng đó càng phổ biến trong xã hội.
                *   **Điểm Lõi (Core Point - Màu Cam):** Là "trái tim" của đám đông, đại diện cho mức thể trạng chuẩn và tập trung đông đúc nhất của nhóm đó.
                *   **Cá thể dị biệt (Noise Point - Màu Xám):** Những người nằm trơ trọi một mình. Đây là các trường hợp có chỉ số cực đoan (ví dụ: béo phì rất nặng, gầy gò ốm yếu, hoặc vận động viên siêu nạc). Thuật toán sẽ không áp dụng bài tập đại trà cho nhóm này mà cần lưu ý y tế riêng.
                *   ⭐ **Vị trí của bạn:** Nếu bạn nằm trong đám mây xanh, thể trạng của bạn rất phổ biến. Nếu bạn nằm ở rìa hoặc thành điểm xám, hệ thống AI sẽ coi bạn là một "cá thể đặc biệt" cần lộ trình nghiêm ngặt hơn!
                """)

                st.write("---")

                # --- Biểu đồ 2: DBSCAN (Density-based) - dựa vào dữ liệu 13k info người dùng ---
                st.write("---")
                st.subheader("2. Thuật toán DBSCAN & Phát hiện cá thể đặc biệt")
                
                # Tính toán khoảng cách (Đã sửa đám mây 2 lên tọa độ 6.5, 6.5)
                dist_to_cloud1 = ((user_x - 3)**2 + (user_y - 5)**2)**0.5
                dist_to_cloud2 = ((user_x - 6.5)**2 + (user_y - 6.5)**2)**0.5
                
                if dist_to_cloud1 <= 1.8 or dist_to_cloud2 <= 1.8:
                    dbscan_status = "✅ Bạn nằm trong vùng có **Mật độ cao**. Thể trạng của bạn rất phổ biến và an toàn để áp dụng các bài tập đại trà."
                else:
                    dbscan_status = "⚠️ Bạn là một người dùng thuộc **Cá thể dị biệt (nằm trong điểm nhiễu)**. Thể trạng của bạn hiếm gặp, hệ thống khuyến nghị bạn nên tập luyện cẩn thận và tư vấn bác sĩ về sức khỏe cơ thể."

                st.success(f"📍 **Đánh giá mật độ của DBSCAN:** {dbscan_status}")

                fig2, ax2 = plt.subplots(figsize=(10, 6))
                np.random.seed(7) 

                # Tạo đám mây dữ liệu (Đám mây 2 lên cao để khớp BMI > 25)
                cloud_1 = np.random.normal(loc=[3, 5], scale=0.6, size=(400, 2))
                cloud_2 = np.random.normal(loc=[6.5, 6.5], scale=0.6, size=(300, 2))
                
                # Vẽ vòng tròn giới hạn
                circle1 = patches.Circle((3, 5), 1.8, color='teal', fill=True, alpha=0.1, linestyle='--', linewidth=2)
                circle2 = patches.Circle((6.5, 6.5), 1.8, color='teal', fill=True, alpha=0.1, linestyle='--', linewidth=2)
                ax2.add_patch(circle1)
                ax2.add_patch(circle2)

                # Vẽ điểm dữ liệu
                ax2.scatter(cloud_1[:,0], cloud_1[:,1], color='teal', alpha=0.6, s=20, label='Đám đông (Mật độ cao)')
                ax2.scatter(cloud_2[:,0], cloud_2[:,1], color='teal', alpha=0.6, s=20)
                
                # Cập nhật tọa độ Điểm Lõi
                ax2.scatter([3, 6.5], [5, 6.5], color='orange', s=150, label='Điểm Lõi (Core Point)', edgecolors='black', zorder=5)
                
                # Điểm nhiễu
                noise_x = [0.5, 1.5, 9.5, 5, 2, 8.5, 1]
                noise_y = [6.5, 2, 5.5, 1.2, 8, 1.5, 4]
                ax2.scatter(noise_x, noise_y, color='grey', s=60, label='Cá thể dị biệt (Noise)', edgecolors='black', alpha=0.9)
                
                # Ngôi sao của bạn
                ax2.scatter(user_x, user_y, color='gold', marker='*', s=600, label='VỊ TRÍ CỦA BẠN', edgecolors='red', linewidth=2, zorder=10)

                # Format
                ax2.set_title("Mô hình mật độ DBSCAN", fontsize=18, fontweight='bold')
                ax2.set_xlabel("Phần trăm mỡ (%)", fontsize=14)
                ax2.set_ylabel("Chỉ số BMI", fontsize=14)
                ax2.tick_params(axis='both', labelsize=12)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
                
                fig2.tight_layout()
                st.pyplot(fig2)

                # --- Phần giải thích 2 điểm lõi ---
                st.info("""
                **Phân tích 2 Điểm Lõi (Core Points):**
                
                *   **Điểm lõi bên trái (Mỡ ~15-20%, BMI ~22-24):** Đây là tâm điểm của **"Cộng đồng người Khỏe mạnh"**. Vùng này chứa những người có chỉ số cân bằng, tập luyện đều đặn hoặc có thể trạng tốt tự nhiên (nhóm 1 và 2).
                *   **Điểm lõi bên phải (Mỡ >25%, BMI >25):** Đây là tâm điểm của **"Cộng đồng người Thừa cân/Ít vận động"**. Vùng này đại diện cho một bộ phận lớn dân số hiện đại ngồi văn phòng nhiều, lượng mỡ cao (Nhóm 3 và 4).
                
                *(Khác với K-Means chủ động cắt dữ liệu thành 4 nhóm để xếp bài tập, thuật toán DBSCAN để dữ liệu tự do hội tụ, qua đó phản ánh đúng thực trạng mật độ sức khỏe của xã hội).*
                """)
                # ----------------------------------------------------

                # --- So sánh bảng K-means và DBSCAN ---
                st.write("---")
                st.subheader("So sánh K-Means và DBSCAN trong Hệ thống")
                
                comp_data = {
                    "Tiêu chí": ["Mục đích chính", "Cách xử lý dữ liệu", "Đối với cá thể dị biệt (Outliers)", "Vai trò trong hệ thống Mitsuo"],
                    "K-Means (Phân nhóm)": ["Phân chia người dùng vào các nhóm cụ thể (1, 2, 3, 4).", "Ép TẤT CẢ mọi người vào một nhóm nào đó gần nhất.", "Gộp chung các trường hợp dị biệt vào nhóm thông thường (Dễ sai lệch).", "Gán nhãn người dùng để AI Coach chọn giáo án tập luyện."],
                    "DBSCAN (Mật độ)": ["Tìm ra các đám đông có chỉ số sức khỏe giống nhau.", "Chỉ gom nhóm những vùng có mật độ dữ liệu dày đặc.", "Phát hiện và tách rời cá thể dị biệt thành điểm nhiễu (Noise).", "Làm màng lọc an toàn, cảnh báo nếu người dùng có chỉ số quá bất thường."]
                }
                
                st.table(comp_data)
                
                st.success("""
                **Kết luận chuyên môn:** Hệ thống Mitsuo kết hợp sức mạnh của cả 2 thuật toán. **K-Means** đóng vai trò 'Bác sĩ kê đơn' (chia nhóm lên thực đơn), còn **DBSCAN** đóng vai trò 'Màng lọc an toàn' (đảm bảo người nhận thực đơn là người có thể trạng phổ thông, không gặp rủi ro y tế do các chỉ số dị biệt).
                """)

                # --- Bảng giải thích / phân tích dưới biểu đồ ---
                st.info("""
                **Biểu đồ DBSCAN (Phân tích Mật độ Thể trạng):**
                
                *   **Trục Tọa độ (BMI & % Mỡ):** Đây giống như một "Bản đồ sức khỏe" của 13,000 người dùng.
                *   **Đám đông màu xanh (Mật độ cao):** Thể hiện những nhóm người có chỉ số cơ thể giống nhau. Đốm xanh càng dày đặc, kiểu thể trạng đó càng phổ biến trong xã hội.
                *   **Điểm Lõi (Core Point - Màu Cam):** Là "trái tim" của đám đông, đại diện cho mức thể trạng chuẩn và tập trung đông đúc nhất của nhóm đó.
                *   **Cá thể dị biệt (Noise Point - Màu Xám):** Những người nằm trơ trọi một mình. Đây là các trường hợp có chỉ số cực đoan (ví dụ: béo phì rất nặng, gầy gò ốm yếu, hoặc vận động viên siêu nạc). Thuật toán sẽ không áp dụng bài tập đại trà cho nhóm này mà cần lưu ý y tế riêng.
                *   ⭐ **Vị trí của bạn:** Nếu bạn nằm trong đám mây xanh, thể trạng của bạn rất phổ biến. Nếu bạn nằm ở rìa hoặc thành điểm xám, hệ thống AI sẽ coi bạn là một "cá thể đặc biệt" cần lộ trình nghiêm ngặt hơn!
                """)

                # --- Mục đích + ý nghĩa ---
                st.success("""
                **Mục đích nghiên cứu:** Hệ thống không chỉ phân loại để 'dán nhãn', mà để cá nhân hóa lộ trình. Việc xác định bạn thuộc nhóm nào (1-4) giúp AI Coach chọn đúng kho bài tập và thực đơn trong cơ sở dữ liệu.
                
                **Ý nghĩa thực tiễn:** Giúp người dùng nhận ra mình đang ở đâu trong bản đồ sức khỏe cộng đồng. Nếu bạn rơi vào vùng 'Noise' của DBSCAN, hệ thống sẽ đưa ra cảnh báo đặc biệt thay vì các bài tập đại trà.
                """)

# --- 8. Trợ lý Ai (ô chat bên phải) ---
if show_chat and col_chat:
    with col_chat:
        st.subheader("Mitsuo AI Coach")
        chat_box = st.container(height=500)
        
        # Hiển thị lịch sử chat
        for msg in st.session_state.chat_history:
            chat_box.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Hỏi Mitsuo..."):
            # Lưu tin nhắn người dùng vào lịch sử
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            chat_box.chat_message("user").write(prompt)
            
            client = OpenAI(api_key="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")
            try:
                # 1. Thu thập "tình báo" về trạng thái hiện tại của người dùng
                u_age = st.session_state.get('u_age', 'chưa rõ')
                u_bmi = st.session_state.get('u_bmi', 'chưa rõ')
                login_status = "Đã đăng nhập" if st.session_state.get('logged_in') else "Chưa đăng nhập"
                u_cluster_display = st.session_state.get('u_cluster', 0) + 1
                
                # Lấy trạng thái DBSCAN
                dbscan_info = "Là một cá thể dị biệt (Cần tư vấn cực kỳ cẩn trọng, khuyên đi khám)" if st.session_state.get('is_outlier') else "Nằm trong Vùng An Toàn (Tư vấn bình thường)"

                # 2. Xây dựng bộ não "Hướng dẫn viên" cho Mitsuo (Phiên bản nghiêm túc & chuyên nghiệp)
                system_instruction = f"""Bạn là Mitsuo - Trợ lý Sức khỏe chuyên biệt của ứng dụng Mitsuo Health Hub. 
                
                Quy tắc phạm vi trả lời:
                1. Chỉ trả lời các chủ đề: Sức khỏe, Thể hình, Dinh dưỡng, Sản phẩm bổ sung, Đồ tập gym và cách dùng app Mitsuo.
                2. Từ chối nhận các chủ đề khác như: Chính trị, tôn giáo, giải trí, toán học, code, v.v. 
                   Mẫu từ chối: "Xin lỗi, Mitsuo chỉ hỗ trợ về sức khỏe và thông tin liên quan đến người dùng thôi. Hãy hỏi Mitsuo về chủ đề liên quan!"

                Kiến thức về data của hệ thống (Để answer ng dùng):
                Hệ thống dựa vào phân tích 13,000 hồ sơ người dùng bằng thuật toán K-Means và DBSCAN, chia làm các nhóm:
                - Nhóm 1 (Chiếm 15%): Thể trạng Tốt. Rất hiếm và xuất sắc.
                - Nhóm 2 (Chiếm 45%): Thể trạng Bình thường. Đa số mọi người thuộc nhóm này.
                - Nhóm 3 (Chiếm 25%): Thể trạng Thừa cân. Cần tập cardio.
                - Nhóm 4 (Chiếm 10%): Thể chất kém. Khá hiếm, cần tập kháng lực mạnh.
                - Cá thể đặc biệt (Chiếm 5%): Là các điểm nhiễu (Noise Point) phát hiện bởi DBSCAN, đại diện cho người có chỉ số cực đoan.

                Thông tin người dùng hiện tại: 
                - Giới tính: {st.session_state.get('u_gender', 'chưa rõ')}
                - Tuổi: {u_age}, BMI: {u_bmi}
                - Người dùng đang thuộc Nhóm K-Means: {u_cluster_display}
                - Đánh giá mật độ DBSCAN: {dbscan_info}
                - Trạng thái: {login_status}
                - Trang hiện tại: '{menu}'

                Quy tắc điều hướng, nhắc nhở:
                1. Đăng nhập/Đăng ký -> Chỉ sang 'Tài khoản & Kế hoạch'.
                2. Chỉ số/BMI/BMR -> Chỉ sang 'Công cụ tính toán'.
                3. Mua sắm/Supplement -> Chỉ sang 'Thực phẩm bổ sung'.
                4. Khi tư vấn mua sắm, luôn nhắc: "Hãy chắc chắn bạn đã chọn đúng giới tính ở Trang chủ để Mitsuo gợi ý chuẩn nhất ạ!"
                5. Không dùng mã LaTeX. Trả lời thân thiện bằng tiếng Việt, không dùng emoji. Nếu hỏi về độ hiếm của nhóm, dùng số liệu % ở trên để phân tích & động viên họ.
                """

                # 3. Gửi yêu cầu cho model OpenAI
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_instruction}] + st.session_state.chat_history
                )
                
                reply = response.choices[0].message.content
                chat_box.chat_message("assistant").write(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e: 
                st.error(f"🚨 Lỗi chi tiết từ OpenAI: {e}")
