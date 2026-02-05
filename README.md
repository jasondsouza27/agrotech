# AgroSmart - Leaf Doctor Module 🌿

> AI-powered crop disease detection with organic remedies

## Overview

The **Leaf Doctor** module provides farmers with an intelligent scanning system to detect crop diseases and nutrient deficiencies. All recommended remedies are **100% organic and sustainable** (Green Growth certified).

## Backend API

### Quick Start

```bash
# Install dependencies
pip install flask flask-cors

# Run the server
python prediction_server.py
```

The server will start at `http://localhost:5000`

### API Endpoints

#### Health Check
```
GET /api/health
```

#### Scan Crop Image
```
POST /api/scan/image
Content-Type: multipart/form-data

Body: image (file)
```

**Response Example:**
```json
{
  "success": true,
  "scan_id": "SCAN-12345",
  "result": {
    "diagnosis": "Nitrogen Deficiency",
    "description": "Yellowing detected in older leaves...",
    "remedy": "Plant beans or legumes nearby to fix nitrogen naturally...",
    "severity": "moderate",
    "confidence": 0.87
  }
}
```

### Possible Diagnoses

| Condition | Severity | Organic Remedy |
|-----------|----------|----------------|
| ✅ Healthy | None | Maintain 60% soil moisture |
| ⚠️ Nitrogen Deficiency | Moderate | Plant legumes, apply compost tea |
| 🔴 Early Blight | High | Neem oil spray, improve air circulation |

## Tech Stack

- **Backend**: Flask (Python)
- **AI Model**: Mock implementation (TensorFlow integration pending)
- **Philosophy**: Green Growth - Sustainable & Organic remedies only

---

*Built for AgroSmart Hackathon 2026*