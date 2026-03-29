from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import datetime

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'consolidated_ink_key_model.pkl')
LOG_PATH = os.path.join(BASE_DIR, 'print_logs.xlsx')

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_all', methods=['POST'])
def predict_all():
    if not model: return jsonify({"status": "error", "message": "Model not loaded"})
    try:
        req = request.json
        results = []
        for z in req['zones']:
            feat = pd.DataFrame([{
                'Color': {'Cyan':0,'Magenta':1,'Yellow':2,'Black':3}.get(z['color'],0),
                'Paper type': {'Coated':0,'Uncoated':1}.get(z['paper_type'],0),
                'Zone number': int(z['zone_no']),
                'Ink key zero setting': float(req['zero_setting']),
                'Delta E improvement': float(z['de_before']) - float(req['target_de']),
                'initial density': 1.31,
                'initial ink key setting': float(z['init_key'])
            }])
            pred = round(float(model.predict(feat)[0]), 2)
            results.append({"zone_no": z['zone_no'], "predicted_key": pred})
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
                'Color': e['color'], 'Paper type': e['paper_type'], 'Zone number': int(e['zone_no']),
                'Ink key zero setting': float(data['zero_setting']), 'Delta E before': float(e['de_before']),
                'Delta E after (Target)': float(data['target_de']), 'initial density': 1.31,
                'initial ink key setting': float(e['init_key']), 'final ink key setting (ACTUAL)': float(e['actual_key']),
                'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            })
        
        df_new = pd.DataFrame(new_entries)
        if os.path.exists(LOG_PATH):
            df_old = pd.read_excel(LOG_PATH)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new.to_excel(LOG_PATH, index=False)
        
        return jsonify({"status": "success", "message": "Actuals saved!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=False)