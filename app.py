import streamlit as st  # Import steht ganz oben
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. Seite konfigurieren
st.set_page_config(page_title="Customer Segment Analysis", layout="wide")

# 2. Ressourcen laden & Reinigung (exakt wie in Ihrem Notebook)
@st.cache_resource
def load_and_clean_data():
    model = joblib.load('models/kmeans_model.pkl')
    try:
        df = pd.read_parquet('data/processed/featrures.parquet')
        # Tippfehler-Korrektur
        df.rename(columns={'latest_redeem_dayes': 'latest_redeem_days'}, inplace=True)
        
        # IQR Reinigung laut Notebook Cell 17
        cols = ['total_redeem_value', 'total_redeem_points', 'latest_redeem_days']
        Q1, Q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
        IQR = Q3 - Q1
        df_cleaned = df[~((df[cols] < (Q1 - 1.5 * IQR)) | (df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)].copy()
        
        # Duplikate entfernen laut Cell 20
        df_final = df_cleaned.drop_duplicates().reset_index(drop=True)
        return model, df_final
    except:
        return model, pd.DataFrame()

model, features = load_and_clean_data()

# Cluster-Definitionen
cluster_info = {
    0: {"Name": "Inaktive Kleinnutzer", "Strategie": "Reaktivierung: Gutschein senden."},
    1: {"Name": "Regelmäßige Gelegenheitsnutzer", "Strategie": "Treue-Bonus anbieten."},
    2: {"Name": "VIP-Kunden (Top-Segment)", "Strategie": "Exklusive VIP-Events."},
    3: {"Name": "Treue Bestandskunden", "Strategie": "Upselling-Angebote."},
    4: {"Name": "Gefährdete Kunden", "Strategie": "Dringlichkeits-Aktion."}
}

st.title("👤 Kunden-Segmentierung & Visualisierung")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Eingabe")
    
    # NEU: Maximale Grenzwerte gesetzt
    latest_days = st.number_input(
        "Tage seit letzter Einlösung", 
        min_value=0, 
        max_value=590,  # Limit auf 590 Tage
        value=50
    )
    
    redeem_value = st.number_input(
        "Einlösewert (Value)", 
        min_value=0.0, 
        max_value=950.0, # Limit auf 950
        value=200.0
    )
    
    # Punkte-Berechnung
    redeem_points = redeem_value * 10
    st.info(f"Berechnete Punkte: **{redeem_points}**")
    
    if st.button("Analyse starten"):
        input_data = [[float(redeem_value), float(redeem_points), float(latest_days)]]
        prediction = model.predict(input_data)[0]
        st.session_state['pred'] = prediction
        st.session_state['input'] = input_data

# 3. Ergebnis & Grafik-Logik
if 'pred' in st.session_state:
    prediction = st.session_state['pred']
    info = cluster_info[prediction]
    
    with col2:
        st.subheader(f"Ergebnis: {info['Name']} (Cluster {prediction})")
        st.success(f"💡 **Marketing-Strategie:** {info['Strategie']}")
        
        if not features.empty:
            st.divider()
            with st.spinner("Grafik wird generiert..."):
                cols = ['total_redeem_value', 'total_redeem_points', 'latest_redeem_days']
                X = features[cols]
                
                # PCA Logik wie im Notebook
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(X)
                
                # Neuen Punkt transformieren
                new_point_pca = pca.transform(pd.DataFrame(st.session_state['input'], columns=cols))
                
                # Plotting
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.scatterplot(
                    x=pca_data[:, 0], y=pca_data[:, 1], 
                    hue=model.predict(X), 
                    palette='tab10', alpha=0.5, s=60, ax=ax
                )
                
                # Das rote X für den neuen Kunden
                ax.scatter(
                    new_point_pca[0, 0], new_point_pca[0, 1], 
                    c='red', marker='X', s=350, 
                    label='Dieser Kunde', edgecolor='black', zorder=15
                )
                
                # Dynamische Achsenanpassung für Sichtbarkeit
                all_x = list(pca_data[:, 0]) + [new_point_pca[0, 0]]
                all_y = list(pca_data[:, 1]) + [new_point_pca[0, 1]]
                ax.set_xlim(min(all_x) - 50, max(all_x) + 50)
                ax.set_ylim(min(all_y) - 50, max(all_y) + 50)
                
                ax.set_title("Kundenposition im Vergleich zu den Clustern")
                ax.legend(title="Cluster-ID", bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)
