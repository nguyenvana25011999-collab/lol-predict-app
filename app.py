import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer

# Giao diện
st.set_page_config(page_title="LOL Pro Stats & Predict", layout="wide")
st.title("Thống Kê & Dự Đoán Giải Đấu LOL Chuyên Nghiệp")

# Cache dữ liệu để không tải lại liên tục
@st.cache_data(ttl=86400) # Cache 1 ngày
def fetch_and_train():
    url = "https://lol.fandom.com/api.php"
    params = {
        "action": "cargoquery", "format": "json",
        "tables": "ScoreboardGames=SG",
        "fields": "SG.Tournament, SG.Team1Picks, SG.Team2Picks, SG.Gamelength_Number, SG.Team1Kills, SG.Team2Kills",
        "where": "SG.Tournament LIKE '%LCK%' OR SG.Tournament LIKE '%LPL%' OR SG.Tournament LIKE '%LEC%' OR SG.Tournament LIKE '%MSI%'",
        "limit": 500
    }
    res = requests.get(url, params=params).json()
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

with st.spinner('Đang tải dữ liệu và huấn luyện mô hình...'):
    model_time, model_kills, encoder = fetch_and_train()

# Form nhập liệu dự đoán
st.subheader("Dự đoán trận đấu")
picks_input = st.text_input("Nhập 10 tướng (Cách nhau bằng dấu phẩy):", "Aatrox, Sejuani, Azir, Lucian, Nami, Ornn, Maokai, Sylas, Zeri, Lulu")

if st.button("Phân tích"):
    picks = [x.strip() for x in picks_input.split(',')]
    if len(picks) == 10:
        X_input = encoder.transform([picks])
        pred_time = model_time.predict(X_input)[0]
        pred_kills = model_kills.predict(X_input)[0]
        
        col1, col2 = st.columns(2)
        col1.metric("Dự đoán thời gian trận", f"{pred_time:.1f} phút")
        col2.metric("Dự đoán tổng Kills", f"{pred_kills:.0f} mạng")
    else:
        st.error("Vui lòng nhập chính xác 10 tướng.")
