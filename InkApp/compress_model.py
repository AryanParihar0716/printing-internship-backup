import joblib
import os

# 1. Load the giant 258MB model
print("Loading model... (this might take a few seconds)")
model = joblib.load('consolidated_ink_key_model.pkl')

# 2. Save it with high compression (level 3 is usually enough)
print("Compressing...")
joblib.dump(model, 'consolidated_ink_key_model.pkl', compress=3)

# 3. Check the new size
new_size = os.path.getsize('consolidated_ink_key_model.pkl') / (1024 * 1024)
print(f"✅ Finished! New File Size: {new_size:.2f} MB")

if new_size < 100:
    print("🚀 Perfect! You can now push to GitHub.")
else:
    print("⚠️ Still over 100MB. Try changing compress=3 to compress=9 in the script.")