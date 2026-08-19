import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer
import plotly.express as px
import google.generativeai as genai
from PIL import Image

# Cấu hình trang
st.set_page_config(page_title="LOL Pro Stats & Predict", layout="wide")
st.title("Thống Kê & Dự Đoán Giải Đấu LOL Chuyên Nghiệp")

# ==========================================
# 1. HỆ THỐNG MÔ HÌNH HỌC MÁY (DỰ ĐOÁN TRẬN ĐẤU)
# ==========================================
@st.cache_data(ttl=86400)
def fetch_and_train():
    # Sử dụng Mock Data để chống lỗi Cloudflare
    mock_data = [
        {"Tournament": "LCK", "Team1Picks": "Aatrox,Sejuani,Azir,Lucian,Nami", "Team2Picks": "Ornn,Maokai,Sylas,Zeri,Lulu", "Gamelength_Number": "35", "Team1Kills": "12", "Team2Kills": "8"},
        {"Tournament": "LPL", "Team1Picks": "Renekton,Vi,Ahri,Aphelios,Thresh", "Team2Picks": "Sion,Wukong,Syndra,Jinx,Nautilus", "Gamelength_Number": "28", "Team1Kills": "22", "Team2Kills": "18"},
        {"Tournament": "LEC", "Team1Picks": "Gwen,Lee Sin,LeBlanc,Xayah,Rakan", "Team2Picks": "K'Sante,Viego,Lissandra,Kaisa,Leona", "Gamelength_Number": "32", "Team1Kills": "15", "Team2Kills": "14"},
        {"Tournament": "MSI", "Team1Picks": "Jax,Sejuani,Annie,Lucian,Nami", "Team2Picks": "Gragas,Vi,Ahri,Aphelios,Thresh", "Gamelength_Number": "30", "Team1Kills": "18", "Team2Kills": "15"}
    ]
    df = pd.DataFrame(mock_data * 50)

    df['Team1Picks'] = df['Team1Picks'].apply(lambda x: x.split(',') if pd.notnull(x) else [])
    df['Team2Picks'] = df['Team2Picks'].apply(lambda x: x.split(',') if pd.notnull(x) else [])
    df['TotalKills'] = pd.to_numeric(df['Team1Kills'].fillna(0)) + pd.to_numeric(df['Team2Kills'].fillna(0))
    df['Gamelength_Number'] = pd.to_numeric(df['Gamelength_Number'].fillna(0))
    
    df['Tournament_Clean'] = df['Tournament']
    df['Features'] = df['Team1Picks'] + df['Team2Picks'] + df['Tournament_Clean'].apply(lambda x: [x])
    
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['Features'])
    rf_time = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['Gamelength_Number'])
    rf_kills = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['TotalKills'])
    return rf_time, rf_kills, mlb

with st.spinner('Đang nạp hệ thống AI...'):
    model_time, model_kills, encoder = fetch_and_train()

# ==========================================
# 2. TÍCH HỢP TÍNH NĂNG NHẬN DIỆN ẢNH BẰNG AI (VÁ LỖI LOCKE)
# ==========================================
st.markdown("---")
st.subheader("📷 Tự động nhận diện tướng từ ảnh Ban/Pick")

uploaded_file = st.file_uploader("Tải ảnh chụp màn hình cấm chọn", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh đang phân tích", use_container_width=True)
    
    if st.button("Quét và Nhận diện ảnh"):
        with st.spinner("AI thị giác đang phân tích khuôn mặt tướng..."):
            try:
                # Lấy API Key từ kho bảo mật của Streamlit
                genai.configure(api_key=st.secrets["GOOGLE_API_KEY"]) 
                model = genai.GenerativeModel('gemini-3.7-flash')
                
                # Prompt đã được nâng cấp bối cảnh để tránh nhầm Ezreal thành Locke
                prompt = "Đây là giao diện cấm chọn giải đấu LOL. Hãy nhìn vào 10 vị tướng được chọn (khung to nhất). Liệt kê tên tiếng Anh chuẩn của 10 vị tướng này, cách nhau bằng dấu phẩy. Chỉ trả về chuỗi văn bản, không giải thích. Chú ý: AI thường nhầm lẫn vị tướng ở đội 2 (Peyz) là Ezreal, nhưng nếu thấy tóc vàng, kính và áo vest xanh thì đó là vị tướng tên 'Locke'. Hãy đảm bảo nhận diện chính xác Locke."
                
                response = model.generate_content([prompt, img])
                detected_picks = response.text.strip()
                
                st.success("Nhận diện hoàn tất!")
                st.info(f"**Kết quả:** {detected_picks}")
                st.write("*(Bạn có thể sao chép chuỗi tướng phía trên để dán vào ô nhập liệu bên dưới)*")
            except Exception as e:
                if "429" in str(e) or "Quota" in str(e):
                    st.warning("⏳ Máy chủ Google đang nghẽn (Giới hạn 5 ảnh/phút). Vui lòng đợi 60 giây rồi thử lại!")
                else:
                    st.error(f"Lỗi kết nối API: {e}")

# ==========================================
# 3. GIAO DIỆN PHÂN TÍCH VÀ DỰ ĐOÁN (BẢNG 2 CỘT)
# ==========================================
st.markdown("---")
st.subheader("Dự đoán trận đấu")
tournament_input = st.selectbox("Chọn giải đấu:", ["LCK", "LPL", "LEC", "MSI"])
picks_input = st.text_input("Nhập 10 tướng (Cách nhau bằng dấu phẩy):", "Jax, Rell, Xin Zhao, Jhin, Orianna, Camille, Locke, Vi, Olaf, Kai'Sa")

if st.button("Phân tích"):
    picks = [x.strip() for x in picks_input.split(',')]
    if len(picks) == 10:
        
        # Tạo bảng 2 cột đối chiếu Đội 1 vs Đội 2
        st.markdown("### 📋 Đội hình thi đấu")
        roles = ["Top (Đường trên)", "Jungle (Đi rừng)", "Mid (Đường giữa)", "ADC (Xạ thủ)", "Support (Hỗ trợ)"]
        df_lineup = pd.DataFrame({
            "Vị trí": roles,
            "Đội 1 (Blue Side)": picks[:5],
            "Đội 2 (Red Side)": picks[5:]
        })
        st.table(df_lineup)
        
        # Chạy dự đoán
        features = picks + [tournament_input]
        # Xử lý các tướng mới (như Locke) chưa có trong dữ liệu huấn luyện
        # Bỏ qua các tướng lạ để tránh lỗi biến đổi (transform)
        valid_features = [f for f in features if f in encoder.classes_]
        X_input = encoder.transform([valid_features])
        
        pred_time = model_time.predict(X_input)[0]
        pred_kills = model_kills.predict(X_input)[0]
        
        st.markdown("### 📊 Kết quả dự đoán")
        col1, col2 = st.columns(2)
        col1.metric("Dự đoán thời gian", f"{pred_time:.1f} phút")
        col2.metric("Dự đoán tổng Kills", f"{pred_kills:.0f} mạng")
    else:
        st.error("Lỗi: Vui lòng nhập chính xác 10 tướng.")

# ==========================================
# 4. BIỂU ĐỒ TRỰC QUAN HÓA LỐI CHƠI (SCATTER PLOT)
# ==========================================
st.markdown("---")
st.subheader("Phân tích phong cách thi đấu các đội (Meta Radar)")

@st.cache_data(ttl=86400)
def fetch_team_playstyle():
    # Mở rộng dữ liệu thống kê đa khu vực
    data = {
        "Team": ["T1", "GEN", "JDG", "BLG", "G2", "FNC", "LNG", "WBG", "DK", "KT", 
                 "GAM", "TW", "C9", "FLY", "HLE", "TES", "MAD", "BDS"],
        "Region": ["LCK", "LCK", "LPL", "LPL", "LEC", "LEC", "LPL", "LPL", "LCK", "LCK", 
                   "VCS", "VCS", "LCS", "LCS", "LCK", "LPL", "LEC", "LEC"],
        "CKPM": [0.65, 0.58, 0.85, 0.92, 0.75, 0.78, 0.70, 0.82, 0.60, 0.68, 
                 0.88, 0.80, 0.62, 0.65, 0.61, 0.89, 0.77, 0.72], 
        "GPM": [1950, 1980, 1920, 1850, 1790, 1750, 1890, 1820, 1880, 1900, 
                1700, 1720, 1810, 1800, 1910, 1880, 1760, 1740]   
    }
    return pd.DataFrame(data)

with st.spinner('Đang tính toán chỉ số CKPM và GPM...'):
    df_teams = fetch_team_playstyle()

fig = px.scatter(
    df_teams, 
    x="GPM", 
    y="CKPM", 
    color="Region", 
    text="Team",
    title="Bản đồ Chiến thuật: Lối chơi Giao tranh vs Kiểm soát tài nguyên",
    labels={
        "GPM": "Lượng vàng thu thập mỗi phút (Farm/Kiểm soát)", 
        "CKPM": "Tổng số mạng mỗi phút (Giao tranh/Combat)"
    }
)

fig.update_traces(textposition='top center', marker=dict(size=12))
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)
