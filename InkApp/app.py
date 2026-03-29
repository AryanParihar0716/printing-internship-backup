from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import datetime
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'consolidated_ink_key_model.pkl')
LOG_PATH = os.path.join(BASE_DIR, 'print_logs.xlsx')

# These MUST match your training script exactly
PAPER_MAP = {'Coated': 0, 'Uncoated': 1}
COLOR_MAP = {'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3}
FEATURES = ['Color', 'Paper type', 'Zone number', 'Ink key zero setting', 'Delta E improvement', 'initial density', 'initial ink key setting']

model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"✅ AI ENGINE ONLINE: Loaded {type(model).__name__}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print("⚠️ PREDICTION DISABLED: 'consolidated_ink_key_model.pkl' not found. Run your trainer script first!")

# Initial load
load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_all', methods=['POST'])
def predict_all():
    if model is None:
        return jsonify({"status": "error", "message": "Model file missing. Please run the trainer script."})
    
    try:
        data = request.json
        zones_input = data.get('zones', [])
        target_de = float(data.get('target_de', 2.5))
        zero_set = float(data.get('zero_setting', 0.0))
        results = []

        for zone in zones_input:
            # Calculate the "Improvement" feature the model expects
            de_improvement = float(zone['de_before']) - target_de
            init_key = float(zone['init_key'])
            init_dens = 1.31 # Standard press baseline

            # Create the data row for the AI
            input_df = pd.DataFrame([{
                'Color': COLOR_MAP.get(zone['color'], 0),
                'Paper type': PAPER_MAP.get(zone['paper_type'], 0),
                'Zone number': int(zone['zone_no']),
                'Ink key zero setting': zero_set,
                'Delta E improvement': de_improvement,
                'initial density': init_dens,
                'initial ink key setting': init_key
            }])

            # Predict
            pred = model.predict(input_df[FEATURES])[0]
            final_key = round(max(0, min(100, float(pred))), 2)
            
            # Physics Bridge for Density (0.3 change per 11% key move)
            dens_delta = (final_key - init_key) * 0.3 / 11
            pred_dens = round(init_dens + dens_delta, 3)

            results.append({
                "zone_no": zone['zone_no'], 
                "predicted_key": final_key, 
                "predicted_density": pred_dens
            })
            
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/save_actuals', methods=['POST'])
def save_actuals():
    try:
        data = request.json
        logs = data.get('logs', [])
        if not logs: return jsonify({"status": "error", "message": "No data received."})

        # Prepare new data for the Excel log
        new_entries = []
        for e in logs:
            new_entries.append({
                'Color': e['color'], 
                'Paper type': e['paper_type'], 
                'Zone number': int(e['zone_no']),
                'Ink key zero setting': float(data['zero_setting']), 
                'Delta E before': float(e['de_before']),
                'Delta E after (Target)': float(data['target_de']), 
                'initial density': 1.31,
                'initial ink key setting': float(e['init_key']), 
                'final ink key setting (ACTUAL)': float(e['actual_key']),
                'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            })

        # Append to the Excel log
        df_new = pd.DataFrame(new_entries)
        if os.path.exists(LOG_PATH):
            df_old = pd.read_excel(LOG_PATH)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        
        df_new.to_excel(LOG_PATH, index=False)
        
        # Optional: Auto-reload the model if the trainer script 
        # is set up to overwrite the .pkl file automatically.
        load_model()

        return jsonify({"status": "success", "message": "Data logged successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Force port 5000 and enable debug for easier dev
    app.run(debug=True, port=5000)