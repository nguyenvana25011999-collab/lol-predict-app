import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer

# Giao diện
st.set_page_config(page_title="LOL Pro Stats & Predict", layout="wide")
st.title("Thống Kê & Dự Đoán Giải Đấu LOL Chuyên Nghiệp")

# Cache dữ liệu
@st.cache_data(ttl=86400)
def fetch_and_train():
    url = "https://lol.fandom.com/api.php"
    # Bổ sung Headers để giả lập trình duyệt, vượt qua tường lửa
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    params = {
        "action": "cargoquery", "format": "json",
        "tables": "ScoreboardGames=SG",
        "fields": "SG.Tournament, SG.Team1Picks, SG.Team2Picks, SG.Gamelength_Number, SG.Team1Kills, SG.Team2Kills",
        "where": "SG.Tournament LIKE '%LCK%' OR SG.Tournament LIKE '%LPL%' OR SG.Tournament LIKE '%LEC%' OR SG.Tournament LIKE '%MSI%'",
        "limit": 500
    }
    
    # Gắn headers vào Request
    res = requests.get(url, headers=headers, params=params).json()
    
    # Bắt lỗi nếu API vẫn trả về rỗng
    if 'cargoquery' not in res:
        return None, None, None

    df = pd.DataFrame([item['title'] for item in res['cargoquery']])
    
    df['Team1Picks'] = df['Team1Picks'].apply(lambda x: x.split(',') if pd.notnull(x) else [])
    df['Team2Picks'] = df['Team2Picks'].apply(lambda x: x.split(',') if pd.notnull(x) else [])
    df['TotalKills'] = pd.to_numeric(df['Team1Kills'].fillna(0)) + pd.to_numeric(df['Team2Kills'].fillna(0))
    df['Gamelength_Number'] = pd.to_numeric(df['Gamelength_Number'].fillna(0))
    
    df['AllPicks'] = df['Team1Picks'] + df['Team2Picks']
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['AllPicks'])
    
    rf_time = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['Gamelength_Number'])
    rf_kills = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['TotalKills'])
    return rf_time, rf_kills, mlb

with st.spinner('Đang tải dữ liệu API và huấn luyện mô hình học máy...'):
    model_time, model_kills, encoder = fetch_and_train()

# Form nhập liệu
st.subheader("Dự đoán trận đấu")
picks_input = st.text_input("Nhập 10 tướng (Cách nhau bằng dấu phẩy):", "Aatrox, Sejuani, Azir, Lucian, Nami, Ornn, Maokai, Sylas, Zeri, Lulu")

if st.button("Phân tích"):
    if model_time is None:
        st.error("Lỗi mất kết nối với máy chủ dữ liệu Leaguepedia. Vui lòng thử lại sau!")
    else:
        picks = [x.strip() for x in picks_input.split(',')]
        if len(picks) == 10:
            # Bỏ qua các tướng chưa từng xuất hiện trong dữ liệu để tránh lỗi AI
            valid_picks = [p for p in picks if p in encoder.classes_]
            if len(valid_picks) < 10:
                st.warning(f"Cảnh báo: Các tướng chưa có dữ liệu thi đấu chuyên nghiệp: {set(picks) - set(valid_picks)}")
                
            X_input = encoder.transform([picks])
            pred_time = model_time.predict(X_input)[0]
            pred_kills = model_kills.predict(X_input)[0]
            
            col1, col2 = st.columns(2)
            col1.metric("Dự đoán thời gian trận", f"{pred_time:.1f} phút")
            col2.metric("Dự đoán tổng Kills", f"{pred_kills:.0f} mạng")
        else:
            st.error("Vui lòng nhập chính xác 10 tướng.")
