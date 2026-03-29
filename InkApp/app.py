from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import datetime
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)

# --- FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'consolidated_ink_key_model.pkl')
LOG_PATH = os.path.join(BASE_DIR, 'print_logs.xlsx')

# Base training files (used to combine with new logs during retraining)
BASE_FILES = [
    'Plus real (0.3-11%).xlsx',
    'real job dataset ( pakka wala ).xlsx',
    'Sample data for 5mm (34_).xlsx',
    'Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx',
    'Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx'
]

# Mappings
PAPER_MAP = {'Coated': 0, 'Uncoated': 1}
COLOR_MAP = {'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3}

# Global model variable
model = None

def load_global_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ AI Model Loaded into memory.")
    else:
        print("⚠️ Model not found. Please run a predict or save to trigger first training.")

load_global_model()

# --- THE AUTO-RETRAINER ENGINE ---
def trigger_auto_retrain():
    global model
    print("🔄 Auto-Retrain Triggered: Rebuilding the AI brain...")
    
    try:
        all_dfs = []
        # 1. Load Original Base Data
        for f in BASE_FILES:
            full_f = os.path.join(BASE_DIR, f)
            if os.path.exists(full_f):
                df = pd.read_excel(full_f)
                df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after', 
                                   'Initial Density': 'initial density', 'Initial Ink Key Setting': 'initial ink key setting'}, inplace=True)
                all_dfs.append(df)

        # 2. Load the New Live Logs
        if os.path.exists(LOG_PATH):
            df_logs = pd.read_excel(LOG_PATH)
            # Standardize log columns to match training features
            df_logs.rename(columns={'Delta E before': 'Delta E before', 'Delta E after (Target)': 'Delta E after',
                                    'final ink key setting (ACTUAL)': 'final ink key setting'}, inplace=True)
            all_dfs.append(df_logs)

        # 3. Combine and Clean
        data = pd.concat(all_dfs, ignore_index=True).dropna()
        data['Color'] = data['Color'].map(COLOR_MAP)
        data['Paper type'] = data['Paper type'].map(PAPER_MAP)
        if data['Ink key zero setting'].dtype == object:
            data['Ink key zero setting'] = data['Ink key zero setting'].str.replace("mm","",regex=False).astype(float)
        data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

        # 4. Train
        features = ['Color', 'Paper type', 'Zone number', 'Ink key zero setting', 'Delta E improvement', 'initial density', 'initial ink key setting']
        X = data[features]
        y = data['final ink key setting']

        new_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        new_model.fit(X, y)

        # 5. Save and Reload
        joblib.dump(new_model, MODEL_PATH)
        model = new_model
        print("✨ Success: Model retrained and updated live!")
        return True
    except Exception as e:
        print(f"❌ Retrain Error: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_all', methods=['POST'])
def predict_all():
    if model is None: return jsonify({"status": "error", "message": "Model not ready."})
    try:
        data = request.json
        zones_input = data['zones']
        target_de = float(data['target_de'])
        zero_set = float(data['zero_setting'])
        results = []

        for zone in zones_input:
            de_improvement = float(zone['de_before']) - target_de
            init_key = float(zone['init_key'])
            init_dens = 1.31

            features = pd.DataFrame([{
                'Color': COLOR_MAP.get(zone['color'], 0),
                'Paper type': PAPER_MAP.get(zone['paper_type'], 0),
                'Zone number': int(zone['zone_no']),
                'Ink key zero setting': zero_set,
                'Delta E improvement': de_improvement,
                'initial density': init_dens,
                'initial ink key setting': init_key
            }])

            pred = model.predict(features)[0]
            final_key = round(max(0, min(100, float(pred))), 2)
            
            # Physics Bridge for Density
            dens_delta = (final_key - init_key) * 0.3 / 11
            pred_dens = round(init_dens + dens_delta, 3)

            results.append({"zone_no": zone['zone_no'], "predicted_key": final_key, "predicted_density": pred_dens})
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/save_actuals', methods=['POST'])
def save_actuals():
    try:
        data = request.json
        logs = data.get('logs', [])
        if not logs: return jsonify({"status": "error", "message": "No data."})

        new_entries = []
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for e in logs:
            new_entries.append({
                'Color': e['color'], 'Paper type': e['paper_type'], 'Zone number': int(e['zone_no']),
                'Ink key zero setting': float(data['zero_setting']), 'Delta E before': float(e['de_before']),
                'Delta E after (Target)': float(data['target_de']), 'initial density': 1.31,
                'initial ink key setting': float(e['init_key']), 'final ink key setting (ACTUAL)': float(e['actual_key'])
            })

        df_new = pd.DataFrame(new_entries)
        if os.path.exists(LOG_PATH):
            df_old = pd.read_excel(LOG_PATH)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new.to_excel(LOG_PATH, index=False)

        # TRIGGER AUTO-LEARNING
        trigger_auto_retrain()

        return jsonify({"status": "success", "message": "Data saved & AI retrained!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)