# E-commerce Recommendation System

A full-stack e-commerce application combining **React Native (Expo + TypeScript)** for the frontend and **FastAPI (Python)** for the backend. It provides real-time product search, personalized recommendations powered by a **Neural Collaborative Filtering (NCF)** model, and interaction tracking for intelligent retraining.

---

## 🚀 Features

* **Dynamic Product Search:** Search products instantly by name, category, or tags from an SQLite database.
* **Personalized Recommendations:** AI-powered product suggestions using a trained NCF model (TensorFlow/Keras).
* **User Interaction Tracking:** Logs user taps and cart actions into `user_interactions.db` for retraining.
* **Hybrid Recommendation Logic:** Combines AI-based predictions and database filtering for accurate results.
* **Model Retraining:** Supports retraining the NCF model via an API endpoint to adapt to new user interactions.
* **Error & Empty States:** Displays friendly UI alerts (e.g., when there’s no network or product found).

---

## 🧠 Tech Stack

### Frontend (React Native)

* **Framework:** React Native (Expo)
* **Language:** TypeScript
* **State Management:** React Hooks (`useState`, `useEffect`, `useCallback`)
* **Networking:** Axios or Fetch API
* **UI Components:** Custom Components + React Native Paper
* **Navigation:** React Navigation (Stack Navigator)
* **Data Fetching:** TanStack Query (React Query)

### Backend (FastAPI)

* **Framework:** FastAPI (Python)
* **ML Framework:** TensorFlow / Keras
* **Database:** SQLite (`ecommerce.db`, `user_interactions.db`)
* **Async Tasks:** ThreadPoolExecutor for retraining
* **API Docs:** Built-in Swagger UI (`/docs`)

---

## 🏗️ Architecture Overview

```
┌────────────────────────────┐
│ React Native Frontend (UI) │
└───────────────┬────────────┘
                │ REST API
┌───────────────▼────────────┐
│      FastAPI Backend       │
│  - /search                 │
│  - /recommendations        │
│  - /interactions           │
│  - /retrain                │
└───────────────┬────────────┘
                │
    ┌───────────┴────────────┐
    │ SQLite Databases       │
    │  - ecommerce.db        │
    │  - user_interactions.db│
    └───────────┬────────────┘
                │
    ┌───────────▼────────────┐
    │ Trained NCF Model (.h5)│
    │ + JSON Mappings        │
    └────────────────────────┘
```

---

## ⚙️ Prerequisites

* **Node.js (>=16)** and npm or yarn
* **Python 3.8+**
* **Expo CLI** (`npm install -g expo-cli`)
* **SQLite Browser (optional)** for database inspection

---

## 🔧 Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd <project-directory>
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
# Activate
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
# or manually
pip install fastapi uvicorn tensorflow numpy pandas
```

Run the server:

```bash
python main.py
# or
uvicorn main:app --reload
```

The API runs at: **[http://localhost:8000](http://localhost:8000)**

### 3. Frontend Setup

```bash
cd EcommRecommenderRN
npm install
# or
yarn install
```

Start Expo:

```bash
npx expo start
# or web preview
npx expo start --web
```

Change API base URL in the frontend if needed:

```ts
const API_BASE_URL = 'http://<your-local-ip>:8000';
```

---

## 📡 API Overview

### `POST /search`

Fetches products matching a query.

```json
{
  "query": "laptop"
}
```

### `POST /v1/recommendations`

Generates personalized product recommendations.

```json
{
  "user_id": "user0",
  "count": 10,
  "search_query": "phone"
}
```

### `POST /interactions`

Logs user interactions.

```json
{
  "user_id": "user0",
  "item_id": "product_01",
  "type": "tap"
}
```

### `POST /retrain`

Retrains the model based on logged interactions.

---

## 🧩 Database Structure

### `ecommerce.db`

| Column     | Type | Description             |
| ---------- | ---- | ----------------------- |
| id         | TEXT | Product ID              |
| name       | TEXT | Product name            |
| price      | REAL | Product price           |
| category   | TEXT | Product category        |
| image_urls | TEXT | JSON list of image URLs |
| tags       | TEXT | JSON list of tags       |

### `user_interactions.db`

| Column    | Type    | Description            |
| --------- | ------- | ---------------------- |
| id        | INTEGER | Auto ID                |
| user_id   | TEXT    | User ID                |
| item_id   | TEXT    | Product ID             |
| type      | TEXT    | Action type (tap/cart) |
| timestamp | REAL    | Action timestamp       |

---

## 🖼️ Images and Placeholders

* Place your product images inside `src/assets/`.
* For live URLs, use direct links like:

  ```json
  "image_urls": ["https://example.com/images/product1.png"]
  ```
* For local files, reference them in React Native:

  ```tsx
  <Image source={require('../assets/product1.png')} />
  ```

---

## 🧪 Usage Flow

1. **Home Screen:** Displays all available products.
2. **Search:** Type in the search bar to fetch matching items from the backend.
3. **Recommendations:** Press Enter to trigger the NCF recommendation API.
4. **Interactions:** Tap or add to cart — interactions are logged.
5. **Retraining:** Call `/retrain` API to update recommendations based on new data.

---

## 🧰 Troubleshooting

* **Images not showing:** Ensure correct URL (`http://10.0.2.2:8000` for Android emulator).
* **Keyboard hides after Enter:** Set `blurOnSubmit={false}` in search bar input.
* **Model not found:** Confirm model/mappings exist under `models_store/<api_key>/`.

---

## 🧑‍💻 Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push (`git push origin feature/my-feature`)
5. Open a pull request 🎉

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify.

---

## 💡 Acknowledgments

* [FastAPI](https://fastapi.tiangolo.com)
* [TensorFlow](https://www.tensorflow.org)
* [React Native](https://reactnative.dev)
* [Expo](https://expo.dev)
* [SQLite Browser](https://sqlitebrowser.org)

---

**Developed by:** K. Sasi Kumar
**Institution:** VIT Bhopal University
**Year:** 2025
