from flask import Flask, render_template, request, jsonify, redirect
import pandas as pd
import joblib
import os
import datetime

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Syncing filename with your trainer script
MODEL_PATH = os.path.join(BASE_DIR, 'decision_tree_model.pkl')
LOG_PATH = os.path.join(BASE_DIR, 'print_logs.xlsx')

# Global Calibration (Shared across all 32 zones)
config = {"zero_setting": 0.0, "target_de": 2.5}

# Load model once at startup
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.route('/')
def index():
    return render_template('index.html', config=config)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        config["zero_setting"] = float(request.form.get("zero_setting", 0.0))
        config["target_de"] = float(request.form.get("target_de", 2.5))
        return redirect('/')
    return render_template('settings.html', config=config)

@app.route('/predict_all', methods=['POST'])
def predict_all():
    if not model:
        return jsonify({"status": "error", "message": "Model file missing. Run trainer script first."})
    
    try:
        req = request.json
        zones_input = req.get('zones', [])
        results = []

        for z in zones_input:
            # 1. Map to match trainer
            c_val = {'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3}.get(z['color'], 0)
            p_val = {'Coated': 0, 'Uncoated': 1}.get(z['paper_type'], 0)
            
            # 2. Build DataFrame with EXACT order from train_model.py
            feat_df = pd.DataFrame([{
                'Color': c_val,
                'Paper type': p_val,
                'Ink key zero setting': config["zero_setting"],
                'Delta E improvement': float(z['de_before']) - config["target_de"],
                'initial density': float(z['init_dens']),
                'initial ink key setting': float(z['init_key'])
            }])

            # 3. Explicitly define column order
            cols = ['Color', 'Paper type', 'Ink key zero setting', 'Delta E improvement', 'initial density', 'initial ink key setting']
            
            # 4. Predict
            pred = model.predict(feat_df[cols])[0]
            
            results.append({
                "zone_no": z['zone_no'], 
                "predicted_key": round(float(pred), 2)
            })

        return jsonify({"status": "success", "results": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/save_actuals', methods=['POST'])
def save_actuals():
    try:
        data = request.json
        logs = data.get('logs', [])
        new_entries = []
        for e in logs:
            new_entries.append({
                'Color': e['color'], 'Paper type': e['paper_type'], 
                'Ink key zero setting': config["zero_setting"], 
                'Delta E before': float(e['de_before']),
                'Delta E after': config["target_de"], 
                'initial density': float(e['init_dens']),
                'initial ink key setting': float(e['init_key']), 
                'final ink key setting': float(e['actual_key']),
                'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            })
        df_new = pd.DataFrame(new_entries)
        if os.path.exists(LOG_PATH):
            df_old = pd.read_excel(LOG_PATH)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new.to_excel(LOG_PATH, index=False)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)