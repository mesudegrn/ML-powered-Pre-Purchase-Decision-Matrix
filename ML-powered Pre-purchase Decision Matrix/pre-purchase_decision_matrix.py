import logging
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# XGBoost ve Prophet opsiyoneldir: kurulu değilse uygulama hata vermeden ilgili modeli atlar.
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

st.set_page_config(page_title="Karar Matrisi", layout="wide", page_icon="🏭")

st.markdown(
    """
    <style>
    .km-header {
        background: linear-gradient(135deg, #1b2735 0%, #2c3e50 100%);
        padding: 2.1rem 2.4rem;
        border-radius: 8px;
        margin-bottom: 1.6rem;
        border-left: 4px solid #c9922f;
    }
    .km-header h1 {
        color: #f2f2f2;
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0 0 0.5rem 0;
        line-height: 1.3;
    }
    .km-header p {
        color: #aab4c0;
        font-size: 0.96rem;
        margin: 0;
        line-height: 1.55;
        max-width: 70ch;
    }
    </style>
    <div class="km-header">
        <h1>Küresel Emtia Fiyat Dalgalanmaları için Pre-satın Alma Karar Matrisi</h1>
        <p>FRED Producer Price Index verileriyle çeyreklik fiyat tahmini yapan ve beş
        modelden hata payı en düşük olanı seçen karar destek sistemi.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HAMMADDE HARİTASI
# ============================================================

commodities = {
    "Alüminyum": {
        "series_id": "WPU102501", "unit": "Endeks", "mode": "index",
        "note": "ABD PPI proxy (Alüminyum Fabrika Ürünleri)",
    },
    "Rubber": {
        "series_id": "WPU071102", "unit": "Endeks", "mode": "index",
        "note": "ABD PPI proxy (Sentetik Kauçuk)",
    },
    "Stainless Steel": {
        "series_id": "WPU10170502", "unit": "Endeks", "mode": "index",
        "note": "ABD PPI proxy (Çelik Tel, Paslanmaz)",
    },
    "Flat Steel": {
        "series_id": "WPU101707", "unit": "Endeks", "mode": "index",
        "note": "ABD PPI proxy (Soğuk Haddelenmiş Çelik Sac/Şerit)",
    },
    "PA6": {
        "series_id": "WPU06621201", "unit": "Endeks", "mode": "index",
        "note": "ABD PPI proxy (Termoplastik Reçineler, genel)",
    },
    "Virgin Filled PP": {
        "series_id": "WPU06621201", "unit": "Endeks", "mode": "index",
        "note": "Aynı genel termoplastik reçine endeksi",
    },
    "EPDM": {
        "series_id": "WPU07110224", "unit": "Endeks", "mode": "index",
        "note": "ABD PPI proxy (Sentetik Kauçuk)",
    },
}

# ============================================================
# 1. SOL MENÜ VE AYARLAR
# ============================================================
st.sidebar.header("Sistem Ayarları")

api_key = st.sidebar.text_input(
    "FRED API Key",
    type="password",
    help="Ücretsiz key: https://fred.stlouisfed.org/docs/api/api_key.html",
)

start_date = st.sidebar.date_input(
    "Veri Başlangıç Tarihi",
    value=date(2010, 1, 1),
    help="Modelin eğitileceği tarihsel verinin başlangıcı.",
)

st.sidebar.markdown("---")
st.sidebar.header("Depo Kapasitesi")
TOTAL_WAREHOUSE_M2 = 1000.0
with st.sidebar.expander("Depo Kısıtını Değiştirmek İçin Tıklayınız"):
    TOTAL_WAREHOUSE_M2 = st.number_input(
        "Toplam Depo Alanı (m²)",
        min_value=0.0, value=1000.0, step=50.0,
        help="Tüm hammaddeler için ortak, tek bir toplam depolama alanı kısıtıdır.",
    )
st.sidebar.caption(f"Aktif toplam depo kısıtı: {TOTAL_WAREHOUSE_M2:,.0f} m²")

st.sidebar.markdown("---")
st.sidebar.header("Hammadde Bazlı Parametreler")
st.sidebar.caption("Sadece işlem yapmak istediğiniz ürünlerin parametrelerini giriniz. Boş bırakılanlar (0) dikkate alınmaz.")

# Dinamik Sol Menü Girdileri
user_params = {}
for name, info in commodities.items():
    with st.sidebar.expander(f"⚙️ {name}", expanded=False):
        birim = st.selectbox(
            "Birim", ["Ton", "Kg"], key=f"birim_{name}",
            help="Aşağıdaki tüm miktar ve fiyat alanları seçtiğiniz birime göre girilmelidir.",
        )

        if info["mode"] == "index":
            cp = st.number_input(f"Güncel Alış Fiyatınız ($/{birim})", min_value=0.0, value=0.0, step=1.0, key=f"cp_{name}")
        else:
            cp = 0.0  # Bu hammadde doğrudan $ bazlı bir seriden geliyorsa fiyat girişine gerek yoktur.

        sp = st.number_input(f"Depolama Maliyeti ($/{birim})", min_value=0.0, value=0.0, step=0.5, key=f"sp_{name}")
        pq = st.number_input(f"Planlanan Alım ({birim})", min_value=0, value=0, step=100, key=f"pq_{name}")
        space_per_unit = st.number_input(
            f"1 {birim} İçin Gereken Depo Alanı (m²)",
            min_value=0.01, value=1.0, step=0.1, key=f"space_{name}",
            help="Zararsız ek alım önerisi bu değer ve toplam depo kısıtı kullanılarak hesaplanır.",
        )

        user_params[name] = {
            "current_price": cp,
            "stocking_price": sp,
            "planned_qty": pq,
            "space_per_unit": space_per_unit,
            "birim": birim,
        }

MIN_QUARTERS = 12       # 5 modelin güvenilir eğitilmesi için asgari çeyrek sayısı
BACKTEST_HOLDOUT = 4    # walk-forward backtest'te tutulan son çeyrek sayısı
N_LAGS = 4              # Random Forest / XGBoost için geriye bakılan çeyrek sayısı

# ============================================================
# 2. VERİ ÇEKME
# ============================================================
@st.cache_data(ttl=3600)
def fetch_fred_data(series_id, api_key, start_date):
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}"
        f"&observation_start={start_date}&file_type=json"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["observations"])
    df = df[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    df = df.set_index("date").resample("QE").mean().reset_index()
    return df

# ============================================================
# 3. TAHMİN MODELLERİ
# ============================================================
def _lag_matrix(values, n_lags):
    X, y = [], []
    for i in range(n_lags, len(values)):
        X.append(values[i - n_lags:i])
        y.append(values[i])
    return np.array(X), np.array(y)

def forecast_holt_winters(dates, values):
    try:
        m = ExponentialSmoothing(values, trend="add", seasonal=None, initialization_method="estimated")
        f = m.fit()
        return float(f.forecast(1)[0])
    except Exception:
        return None

def forecast_arima(dates, values):
    best_aic, best_pred = np.inf, None
    for d in (1, 0):
        for p in range(3):
            for q in range(3):
                try:
                    f = ARIMA(values, order=(p, d, q)).fit()
                    if np.isfinite(f.aic) and f.aic < best_aic:
                        best_aic, best_pred = f.aic, float(f.forecast(1)[0])
                except Exception:
                    continue
        if best_pred is not None:
            break
    return best_pred

def forecast_random_forest(dates, values, n_lags=N_LAGS):
    if len(values) <= n_lags + 2:
        return None
    X, y = _lag_matrix(values, n_lags)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return float(model.predict(values[-n_lags:].reshape(1, -1))[0])

def forecast_xgboost(dates, values, n_lags=N_LAGS):
    if not XGBOOST_AVAILABLE or len(values) <= n_lags + 2:
        return None
    X, y = _lag_matrix(values, n_lags)
    model = xgb.XGBRegressor(n_estimators=200, max_depth=3, random_state=42, verbosity=0)
    model.fit(X, y)
    return float(model.predict(values[-n_lags:].reshape(1, -1))[0])

def forecast_prophet(dates, values):
    if not PROPHET_AVAILABLE or len(values) < 8:
        return None
    try:
        df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=1, freq="QE")
        fcst = m.predict(future)
        return float(fcst["yhat"].iloc[-1])
    except Exception:
        return None

MODEL_REGISTRY = {
    "Holt-Winters": forecast_holt_winters,
    "ARIMA": forecast_arima,
    "Random Forest": forecast_random_forest,
    "XGBoost": forecast_xgboost,
    "Prophet": forecast_prophet,
}

def _backtest_models(dates_tuple, values_tuple, holdout=BACKTEST_HOLDOUT):
    values = np.array(values_tuple)
    dates = list(dates_tuple)
    n = len(values)
    holdout = min(holdout, max(2, n // 4))
    errors = {name: [] for name in MODEL_REGISTRY}

    for i in range(n - holdout, n):
        train_values, train_dates, actual = values[:i], dates[:i], values[i]
        if len(train_values) < 6:
            continue
        for name, fn in MODEL_REGISTRY.items():
            try:
                pred = fn(train_dates, train_values)
                if pred is None or not np.isfinite(pred):
                    continue
                ape = abs((actual - pred) / actual) if actual != 0 else np.nan
                if np.isfinite(ape):
                    errors[name].append(ape)
            except Exception:
                continue
    return {name: float(np.mean(e)) * 100 for name, e in errors.items() if e}

@st.cache_data(ttl=3600)
def best_model_forecast(dates_tuple, values_tuple):
    mape = _backtest_models(dates_tuple, values_tuple)
    values = np.array(values_tuple)
    dates = list(dates_tuple)

    if not mape:
        pred = forecast_holt_winters(dates, values)
        return pred, "Holt-Winters (fallback)", {}

    for name in sorted(mape, key=mape.get):
        pred = MODEL_REGISTRY[name](dates, values)
        if pred is not None and np.isfinite(pred):
            return pred, name, mape

    return float(values[-1]), "Son Değer (fallback)", mape

# ============================================================
# 4. ANA ÇALIŞMA ALANI
# ============================================================
if not api_key:
    st.warning("Lütfen sol menüden FRED API Key'inizi giriniz. Ücretsiz key için fred.stlouisfed.org adresini ziyaret edebilirsiniz.")
    st.stop()

if not XGBOOST_AVAILABLE:
    st.sidebar.caption("⚠️ xgboost kurulu değil, o model atlanacak (`pip install xgboost`).")
if not PROPHET_AVAILABLE:
    st.sidebar.caption("⚠️ prophet kurulu değil, o model atlanacak (`pip install prophet`).")

all_data = []
predictions_list = []
model_reports = {}
errors = []

progress = st.progress(0.0, text="Veri çekiliyor ve modeller eğitiliyor. Lütfen bekleyiniz.")

for idx, (name, info) in enumerate(commodities.items()):
    try:
        df = fetch_fred_data(info["series_id"], api_key, start_date)
    except requests.exceptions.HTTPError:
        errors.append(f"{name}: API isteği başarısız oldu (key veya series_id hatalı olabilir).")
        progress.progress((idx + 1) / len(commodities))
        continue
    except Exception as e:
        errors.append(f"{name}: Beklenmeyen hata — {e}")
        progress.progress((idx + 1) / len(commodities))
        continue

    if df is None or df.empty:
        errors.append(f"{name}: Veri bulunamadı.")
        progress.progress((idx + 1) / len(commodities))
        continue

    if len(df) < MIN_QUARTERS:
        errors.append(f"{name}: Yalnızca {len(df)} çeyrek veri var, en az {MIN_QUARTERS} çeyrek gerekiyor.")
        progress.progress((idx + 1) / len(commodities))
        continue

    df["Hammadde"] = name
    df["Birim"] = info["unit"]
    all_data.append(df)

    dates_tuple = tuple(df["date"].tolist())
    values_tuple = tuple(df["value"].tolist())
    forecast_val, best_model, mape = best_model_forecast(dates_tuple, values_tuple)
    model_reports[name] = {"best": best_model, "mape": mape}

    current_val = df["value"].values[-1]
    pct_change = (forecast_val - current_val) / current_val * 100 if current_val else np.nan

    # Kullanıcının yan panelden girdiği parametreler alınır.
    params = user_params[name]
    stock_price = params["stocking_price"]
    plan_qty = params["planned_qty"]
    space_per_unit = params["space_per_unit"]
    birim = params["birim"]

    row = {
        "Hammadde": name,
        "Birim": birim,
        "Mod": "$ Hesap Aktif" if (info["mode"] == "usd" or params["current_price"] > 0) else "Sadece Endeks",
        "En İyi Model": best_model,
        "En Düşük Hata (MAPE %)": round(min(mape.values()), 2) if mape else None,
        "Güncel Değer": round(current_val, 2),
        "Tahmin (Gelecek Çeyrek)": round(forecast_val, 2),
        "% Değişim": round(pct_change, 2),
    }

    has_dollar = False
    price_now = None
    net_profit_per_unit = None
    if info["mode"] == "usd":
        has_dollar = True
        price_now, price_next = current_val, forecast_val
    elif params["current_price"] > 0:
        has_dollar = True
        price_now = params["current_price"]
        price_next = price_now * (forecast_val / current_val) if current_val else price_now

    # Değer hesaplanamıyorsa arayüzde "-" gösterilir.
    if has_dollar:
        price_diff = price_next - price_now
        net_profit_per_unit = price_diff - stock_price

        row["Fiyat Değişimi ($)"] = round(price_diff, 2)
        row["Net Kâr ($)"] = round(net_profit_per_unit, 2)
        row["Plan Etkisi ($)"] = round(net_profit_per_unit * plan_qty, 2) if plan_qty > 0 else "-"
        row["Önerilen Aksiyon"] = "ALIMI ARTIR" if net_profit_per_unit > 0 else "STANDART ALIM"
    else:
        row["Fiyat Değişimi ($)"] = "-"
        row["Net Kâr ($)"] = "-"
        row["Plan Etkisi ($)"] = "-"
        row["Önerilen Aksiyon"] = "FİYAT GİRİN"

    # Depo kısıtlı ek alım hesaplaması için gereken ara değerler (tabloda gösterilmez).
    row["_price_now"] = price_now
    row["_net_profit_per_unit"] = net_profit_per_unit
    row["_space_per_unit"] = space_per_unit
    row["_planned_qty"] = plan_qty

    predictions_list.append(row)
    progress.progress((idx + 1) / len(commodities))

progress.empty()
for err in errors:
    st.info(err)

# ============================================================
# DEPO KISITLI, ZARARSIZ EK ALIM HESABI
# ============================================================
# Planlanan alımların kapladığı toplam alan düşülerek kalan kapasite bulunur. Her kalem için
# ayrı ayrı "kalan kapasitenin tamamı bu kaleme ayrılırsa ulaşılabilecek üst sınır" hesaplanır.
# Not: Bu, her kalem için bağımsız bir üst sınırdır; birden fazla kaleme aynı anda bu sınıra
# kadar alım yapılması durumunda toplam alan 1000 m² kısıtını aşabilir.
used_space = sum((row["_planned_qty"] or 0) * (row["_space_per_unit"] or 0) for row in predictions_list)
remaining_space = max(0.0, TOTAL_WAREHOUSE_M2 - used_space)

for row in predictions_list:
    if row["_net_profit_per_unit"] is not None and row["_net_profit_per_unit"] > 0 and row["_space_per_unit"] > 0:
        max_extra = remaining_space / row["_space_per_unit"]
        row["Zararsız Ek Alım Önerisi"] = round(max_extra, 1)
        row["Ek Alımın Finansal Etkisi ($)"] = round(max_extra * row["_net_profit_per_unit"], 2)
    elif row["_net_profit_per_unit"] is not None:
        row["Zararsız Ek Alım Önerisi"] = 0.0
        row["Ek Alımın Finansal Etkisi ($)"] = 0.0
    else:
        row["Zararsız Ek Alım Önerisi"] = "-"
        row["Ek Alımın Finansal Etkisi ($)"] = "-"

# ============================================================
# 5. SONUÇLAR VE ARAYÜZ
# ============================================================
if predictions_list:
    st.subheader("Pre-satın Alma Karar Matrisi Sonuçları")
    results_df = pd.DataFrame(predictions_list)
    display_cols = [c for c in results_df.columns if not c.startswith("_")]
    st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
    st.caption(
        f"Zararsız Ek Alım Önerisi, kalan depo kapasitesinin ({remaining_space:,.1f} m²) tamamının "
        "ilgili kaleme ayrılması varsayımıyla hesaplanmıştır."
    )

    with st.expander("Model Karşılaştırması"):
        for name, report in model_reports.items():
            st.markdown(f"**{name}** — Kazanan: `{report['best']}`")
            if report["mape"]:
                mape_df = pd.DataFrame(
                    sorted(report["mape"].items(), key=lambda x: x[1]),
                    columns=["Model", "MAPE (%)"],
                )
                st.dataframe(mape_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Backtest için yeterli veri/model bulunamadı, fallback tahmin kullanıldı.")

    # --- Aksiyon Önerileri ve Finansal Değerlendirme ---
    st.markdown("### Stratejik Aksiyon Önerileri ve Finansal Değerlendirme")

    for row in predictions_list:
        name = row["Hammadde"]
        birim = row["Birim"]

        if row["Net Kâr ($)"] != "-":
            if row["Net Kâr ($)"] > 0:
                box_html = f"""
                <div style="background-color: rgba(255, 75, 75, 0.15); border-left: 3px solid #ff4b4b; padding: 14px; border-radius: 4px; margin-bottom: 12px;">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; font-weight: 600; font-size: 16px;">{name}</h4>
                        <span style="margin-left: 12px; background-color: #ff4b4b; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;">ALIMI ARTIR</span>
                    </div>
                    <p style="margin: 0 0 4px 0; font-size: 14px; opacity: 0.9;"><b>Analiz:</b> <code>{row['En İyi Model']}</code> modeline göre beklenen fiyat artışı <b>${row['Fiyat Değişimi ($)']:.2f}</b>. Depolama maliyeti düşüldüğünde net avantaj: <b>${row['Net Kâr ($)']:.2f}</b>/{birim}.</p>
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;"><b>Öneri:</b> Kalan depo kapasitesinin tamamı bu kaleme ayrılırsa, zarara girmeden yaklaşık <b>{row['Zararsız Ek Alım Önerisi']:,.1f} {birim}</b> ek alım yapılabilir; tahmini finansal etki <b>${row['Ek Alımın Finansal Etkisi ($)']:,.2f}</b>'dir.</p>
                </div>
                """
                st.markdown(box_html, unsafe_allow_html=True)
            else:
                box_html = f"""
                <div style="background-color: rgba(0, 204, 150, 0.15); border-left: 3px solid #00cc96; padding: 14px; border-radius: 4px; margin-bottom: 12px;">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; font-weight: 600; font-size: 16px;">{name}</h4>
                        <span style="margin-left: 12px; background-color: #00cc96; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;">STANDART PLAN</span>
                    </div>
                    <p style="margin: 0 0 4px 0; font-size: 14px; opacity: 0.9;"><b>Analiz:</b> <code>{row['En İyi Model']}</code> modeline göre öngörülen fiyat hareketi (${row['Fiyat Değişimi ($)']:.2f}), depolama maliyetini karşılamamaktadır.</p>
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;"><b>Öneri:</b> Ek alım önerilmemektedir; mevcut tedarik planına sadık kalınması tavsiye edilir.</p>
                </div>
                """
                st.markdown(box_html, unsafe_allow_html=True)
        else:
            direction = "artış" if row["% Değişim"] > 0 else "düşüş"
            box_html = f"""
            <div style="background-color: rgba(99, 110, 250, 0.15); border-left: 3px solid #636efa; padding: 14px; border-radius: 4px; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <h4 style="margin: 0; font-weight: 600; font-size: 16px;">{name}</h4>
                    <span style="margin-left: 12px; background-color: #636efa; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;">VERİ BEKLENİYOR</span>
                </div>
                <p style="margin: 0 0 4px 0; font-size: 14px; opacity: 0.9;"><b>Analiz:</b> <code>{row['En İyi Model']}</code> modeline göre %{abs(row['% Değişim']):.1f} oranında {direction} öngörülmektedir.</p>
                <p style="margin: 0; font-size: 14px; opacity: 0.9;"><b>Öneri:</b> Finansal kârlılık analizi için sol menüden güncel alış fiyatının girilmesi gerekmektedir.</p>
            </div>
            """
            st.markdown(box_html, unsafe_allow_html=True)

        # --- Belirli bir ek alım miktarının etkisini gösteren simülasyon ---
        if row["_net_profit_per_unit"] is not None:
            sim_qty = st.number_input(
                f"{name} — Belirli Bir Ek Alım Miktarının Finansal Etkisini Görüntüleyin ({birim})",
                min_value=0.0, value=0.0, step=10.0, key=f"sim_{name}",
            )
            if sim_qty > 0:
                sim_profit = row["_net_profit_per_unit"] * sim_qty
                sim_space = sim_qty * row["_space_per_unit"]
                sim_pct = (sim_profit / (row["_price_now"] * sim_qty) * 100) if row["_price_now"] else None
                c1, c2, c3 = st.columns(3)
                c1.metric("Tahmini Finansal Etki", f"${sim_profit:,.2f}")
                c2.metric("Getiri Oranı", f"%{sim_pct:.2f}" if sim_pct is not None else "—")
                c3.metric("Gereken Depo Alanı", f"{sim_space:,.1f} m²")
                if sim_space > remaining_space:
                    st.caption(f"Bu miktar, kalan boş depo kapasitesini ({remaining_space:,.1f} m²) aşmaktadır.")
        st.markdown("")

    st.markdown("---")

    st.subheader("Tarihsel Fiyat / Endeks Trendleri")
    n_cols = 3
    n_rows = -(-len(all_data) // n_cols)
    subplot_titles = [f"{df['Hammadde'].iloc[0]} ({df['Birim'].iloc[0]})" for df in all_data]
    trend_fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
    for i, df in enumerate(all_data):
        r, c = i // n_cols + 1, i % n_cols + 1
        trend_fig.add_trace(
            go.Scatter(x=df["date"], y=df["value"], mode="lines+markers", showlegend=False),
            row=r, col=c,
        )
    trend_fig.update_layout(height=280 * n_rows, margin=dict(t=60, b=20))
    st.plotly_chart(trend_fig, use_container_width=True)

    st.subheader("Önümüzdeki Çeyrek İçin % Değişim Tahmini")
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in results_df["% Değişim"]]
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=results_df["Hammadde"], y=results_df["% Değişim"], marker_color=colors))
    bar_fig.update_layout(yaxis_title="% Değişim", height=400)
    st.plotly_chart(bar_fig, use_container_width=True)

    # ============================================================
    # 6. DETAYLI RAPOR (İNDİRİLEBİLİR)
    # ============================================================
    st.markdown("---")
    st.subheader("Detaylı Model Raporu")
    st.caption(
        "Hangi hammadde için hangi modelin kazandığı, tüm modellerin hata oranları ve "
        "hangi testlerin yapıldığı — bu çalıştırmanın gerçek verisiyle üretilmiştir."
    )

    def build_markdown_report():
        lines = []
        lines.append("# Pre-satın Alma Karar Matrisi — Detaylı Model Raporu")
        lines.append(f"\nOluşturulma zamanı: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
        lines.append(f"\nVeri başlangıcı: {start_date} | Toplam depo kısıtı: {TOTAL_WAREHOUSE_M2:,.0f} m²\n")

        lines.append("## Metodoloji\n")
        lines.append(
            "- Her hammadde için birbirinden bağımsız 5 model denendi: Holt-Winters (statsmodels), "
            "ARIMA (p,d,q ızgara taraması, AIC ile seçim), Random Forest (son 4 çeyrek lag "
            "özellik olarak), XGBoost (aynı lag yapısı)" + (", Prophet" if PROPHET_AVAILABLE else "") + ".\n"
        )
        if not XGBOOST_AVAILABLE:
            lines.append("- xgboost kütüphanesi bu çalıştırmada kurulu değildi, bu model atlandı.\n")
        if not PROPHET_AVAILABLE:
            lines.append("- prophet kütüphanesi bu çalıştırmada kurulu değildi, bu model atlandı.\n")
        lines.append(
            f"- Walk-forward backtest: son {BACKTEST_HOLDOUT} çeyrek sırayla test için ayrıldı. "
            "Her adımda modeller yalnızca o ana kadarki veriyle eğitildi, bir sonraki çeyrek tahmin edildi "
            "ve gerçekleşen değerle karşılaştırıldı.\n"
        )
        lines.append(
            "- Hata metriği: MAPE (Mean Absolute Percentage Error). Yaklaşık doğruluk = 100% − MAPE "
            "olarak raporlanmaktadır; bu, resmi bir doğruluk skoru değil kaba bir yorumlama aracıdır.\n"
        )
        lines.append(
            "- Her hammadde için backtest'teki ortalama MAPE değeri en düşük olan model, tüm veriyle "
            "yeniden eğitilerek gelecek çeyrek tahmini için kullanılmıştır. Belirlenen 'tüm modelleri dene, "
            "en düşük hata oranına sahip olanı kullan' kuralı bu şekilde uygulanmıştır.\n"
        )
        lines.append(
            f"- Depo kısıtı: Toplam depolama alanı {TOTAL_WAREHOUSE_M2:,.0f} m² olarak tanımlanmıştır. "
            f"Planlanan alımların kapladığı alan düşüldükten sonra kalan {remaining_space:,.1f} m², "
            "her kalem için ayrı ayrı 'bu alanın tamamı bu kaleme ayrılırsa ulaşılabilecek üst sınır' "
            "varsayımıyla zararsız ek alım önerisine dönüştürülmüştür.\n"
        )

        lines.append("\n## Hammadde Bazında Sonuçlar\n")
        for row in predictions_list:
            name = row["Hammadde"]
            report = model_reports.get(name, {})
            params = user_params[name]

            lines.append(f"### {name} ({row['Birim']}, {row['Mod']})")
            lines.append(f"Kullanılan FRED serisi: `{commodities[name]['series_id']}` — {commodities[name]['note']}\n")
            lines.append(
                f"Parametreler: Planlanan Alım {params['planned_qty']} {params['birim']} | "
                f"Depolama Maliyeti ${params['stocking_price']:.2f}/{params['birim']} | "
                f"Depo Alanı İhtiyacı {params['space_per_unit']} m²/{params['birim']}\n"
            )

            if report.get("mape"):
                lines.append("| Model | MAPE (%) | Yaklaşık Doğruluk (%) |")
                lines.append("|---|---|---|")
                for model_name, mape_val in sorted(report["mape"].items(), key=lambda x: x[1]):
                    lines.append(f"| {model_name} | {mape_val:.2f} | {100 - mape_val:.2f} |")
                lines.append("")
            else:
                lines.append("Backtest için yeterli veri yoktu, fallback tahmin kullanıldı.\n")

            lines.append(f"Kazanan model: {row['En İyi Model']}")
            lines.append(
                f"Güncel değer: {row['Güncel Değer']} | Tahmin: {row['Tahmin (Gelecek Çeyrek)']} "
                f"| % Değişim: {row['% Değişim']}%"
            )

            if row["Net Kâr ($)"] != "-":
                lines.append(f"Birim başına net kâr: ${row['Net Kâr ($)']:.2f} | Önerilen aksiyon: {row['Önerilen Aksiyon']}")
                lines.append(
                    f"Zararsız ek alım önerisi (depo kısıtına göre): {row['Zararsız Ek Alım Önerisi']:,.1f} {row['Birim']} "
                    f"(tahmini finansal etki: ${row['Ek Alımın Finansal Etkisi ($)']:,.2f})"
                )
            else:
                lines.append(f"Önerilen aksiyon: {row['Önerilen Aksiyon']} (bilinen fiyat girilmedi)")
            lines.append("")

        return "\n".join(lines)

    report_md = build_markdown_report()
    st.download_button(
        "Raporu Markdown Olarak İndir",
        data=report_md,
        file_name=f"karar_matrisi_raporu_{pd.Timestamp.now():%Y%m%d}.md",
        mime="text/markdown",
    )
    with st.expander("Raporu burada önizle"):
        st.markdown(report_md)
else:
    st.error("Hiçbir hammadde için veri/tahmin üretilemedi. Yukarıdaki hata mesajlarını kontrol edin.")
