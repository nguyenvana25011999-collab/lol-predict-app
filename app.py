import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(page_title="LOL Pro Stats & Predict", layout="wide")
st.title("Thống Kê & Dự Đoán Giải Đấu LOL Chuyên Nghiệp")

@st.cache_data(ttl=86400)
def fetch_and_train():
    url = "https://lol.fandom.com/api.php"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0"}
    params = {
        "action": "cargoquery", "format": "json",
        "tables": "ScoreboardGames=SG",
        "fields": "SG.Tournament, SG.Team1Picks, SG.Team2Picks, SG.Gamelength_Number, SG.Team1Kills, SG.Team2Kills",
        "where": "SG.Tournament LIKE '%LCK%' OR SG.Tournament LIKE '%LPL%' OR SG.Tournament LIKE '%LEC%' OR SG.Tournament LIKE '%MSI%'",
        "limit": 500
    }
    res = requests.get(url, headers=headers, params=params).json()
    if 'cargoquery' not in res: return None, None, None

    df = pd.DataFrame([item['title'] for item in res['cargoquery']])
    df['Team1Picks'] = df['Team1Picks'].apply(lambda x: x.split(',') if pd.notnull(x) else [])
    df['Team2Picks'] = df['Team2Picks'].apply(lambda x: x.split(',') if pd.notnull(x) else [])
    df['TotalKills'] = pd.to_numeric(df['Team1Kills'].fillna(0)) + pd.to_numeric(df['Team2Kills'].fillna(0))
    df['Gamelength_Number'] = pd.to_numeric(df['Gamelength_Number'].fillna(0))
    
    # Lọc và đồng nhất tên giải đấu
    def clean_tour(t):
        for main in ["LCK", "LPL", "LEC", "MSI"]:
            if main in str(t): return main
        return str(t)
    df['Tournament_Clean'] = df['Tournament'].apply(clean_tour)
    
    # Gộp Tướng và Tên Giải đấu thành 1 tổ hợp biến để train AI
    df['Features'] = df['Team1Picks'] + df['Team2Picks'] + df['Tournament_Clean'].apply(lambda x: [x])
    
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['Features'])
    
    rf_time = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['Gamelength_Number'])
    rf_kills = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['TotalKills'])
    return rf_time, rf_kills, mlb

with st.spinner('Đang tải dữ liệu API và phân tích meta giải đấu...'):
    model_time, model_kills, encoder = fetch_and_train()

st.subheader("Dự đoán trận đấu")
# Thêm menu chọn giải đấu
tournament_input = st.selectbox("Chọn giải đấu (Để AI điều chỉnh tỷ lệ tính toán):", ["LCK", "LPL", "LEC", "MSI"])
picks_input = st.text_input("Nhập 10 tướng (Cách nhau bằng dấu phẩy):", "Aatrox, Sejuani, Azir, Lucian, Nami, Ornn, Maokai, Sylas, Zeri, Lulu")

if st.button("Phân tích"):
    if model_time is None:
        st.error("Mất kết nối dữ liệu!")
    else:
        picks = [x.strip() for x in picks_input.split(',')]
        if len(picks) == 10:
            # Truyền thêm giải đấu vào mô hình dự đoán
            features = picks + [tournament_input]
            X_input = encoder.transform([features])
            
            pred_time = model_time.predict(X_input)[0]
            pred_kills = model_kills.predict(X_input)[0]
            
            col1, col2 = st.columns(2)
            col1.metric(f"Dự đoán thời gian ({tournament_input})", f"{pred_time:.1f} phút")
            col2.metric(f"Dự đoán tổng Kills ({tournament_input})", f"{pred_kills:.0f} mạng")
        else:
            st.error("Vui lòng nhập chính xác 10 tướng.")
