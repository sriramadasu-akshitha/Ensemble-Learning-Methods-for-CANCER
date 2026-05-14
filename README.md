## MajorProject: Frontend ↔ Backend connection

This project serves `frontend.html` and connects it to a Python prediction API.

### Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start server:

```bash
python model.py
```

Open:
`http://127.0.0.1:5000/`

### API

- **POST** `/predict`
  - Form-data field: `image` (image file)
  - Returns JSON:
    - `label`: `"Normal"` or `"Disease Detected"`
    - `confidence_percent`: 0-100

### Optional: plug in your trained ensemble

If you export your trained artifacts, put them here:

- `models/model1.keras`
- `models/model2.keras`
- `models/feature_extractor.keras`
- `models/rf.pkl`
- `models/svm.pkl`

If these files aren’t present, the backend uses a lightweight fallback heuristic so the UI still works end-to-end.

