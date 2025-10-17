# 🛍️ E-commerce Product Recommender App

An AI-powered hybrid recommendation system built using **React Native**, **FastAPI**, and **Neural Collaborative Filtering (NCF)**. The app recommends products based on user behavior and supports text-based product search with intelligent ranking and hybrid filtering.

---

## 🚀 Features

* 🔍 **Hybrid Recommendations** — Combines collaborative filtering (NCF) with database search.
* 👤 **User-Specific Suggestions** — Personalized predictions using trained models.
* 🧠 **AI-Powered Model** — Uses deep learning for accurate product ranking.
* 💬 **Search Integration** — Filters and ranks recommendations by product search queries.
* 📶 **Offline & Error Handling** — Displays an alert image when there’s no network or data.
* 🖼️ **Dynamic Product Images** — Easily replace placeholders with real product URLs.

---

## 🏗️ Tech Stack

**Frontend:** React Native (Expo)
**Backend:** FastAPI
**Model:** Neural Collaborative Filtering (TensorFlow / Keras)
**Database:** SQLite / PostgreSQL
**API Key Management:** Custom authentication layer in FastAPI

---

## 📦 Installation

### Clone the repository

```bash
git clone https://github.com/KmSasikumar/EcommProductRecommender2.git
cd EcommProductRecommender2
```

### Backend Setup (FastAPI)

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```
2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI server:

   ```bash
   uvicorn main:app --reload
   ```

Your backend will now be available at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

### Frontend Setup (React Native)

1. Navigate to the app folder:

   ```bash
   cd src
   ```
2. Install dependencies:

   ```bash
   npm install
   ```
3. Start the development server:

   ```bash
   npx expo start
   ```
4. Use the QR code to open the app on your device (Expo Go app).

---

## 🖼️ Replacing Placeholder Images

You can replace product placeholder URLs in the product data or response with your actual image URLs.

Example:

```js
{
  id: 1,
  name: "Wireless Headphones",
  price: 49.99,
  image: "https://yourdomain.com/images/headphones.jpg"
}
```

The frontend will automatically display these images instead of the default placeholders.

---

## ⚠️ Network Error / Empty State

When there is **no network connection** or **no product results**, the app shows an alert image located at:

```
D:\Projects\EcommRecommenderRN\src\assets\alert.png
```

This helps maintain a clean user experience even in error conditions.

---

## 🧩 API Overview

### `/recommend`

* **Method:** `POST`
* **Description:** Returns product recommendations for a given user, optionally filtered by a search query.

#### Request Body:

```json
{
  "user_id": "U123",
  "count": 10,
  "search_query": "laptop"
}
```

#### Response:

```json
{
  "user_id": "U123",
  "recommendations": [
    { "item_id": "P001", "score": 0.98 },
    { "item_id": "P002", "score": 0.96 }
  ]
}
```

---

## 🧠 How It Works

1. The backend loads the **trained NCF model** and user/item mappings.
2. The user’s behavior and interactions are converted into feature embeddings.
3. The model predicts the top-N products based on the user’s profile.
4. If a search query is present, the system filters recommendations using DB results.
5. The hybrid engine returns a ranked list of personalized recommendations.

---

## 🧑‍💻 Project Structure

```
EcommProductRecommender2/
│
├── backend/
│   ├── main.py
│   ├── models/
│   ├── utils/
│   ├── requirements.txt
│   └── ...
│
├── src/
│   ├── components/
│   ├── screens/
│   ├── assets/
│   │   └── alert.png
│   ├── App.js
│   └── ...
│
└── README.md
```

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature/new-feature
   ```
3. Commit changes:

   ```bash
   git commit -m "Added new feature"
   ```
4. Push to your branch:

   ```bash
   git push origin feature/new-feature
   ```
5. Create a Pull Request.

---

## 💡 Author

👤 **K. Sasi Kumar**
📧 Email: [kommamani012@gmail.com](mailto:kommamani012@gmail.com)]
🏫 VIT Bhopal University
🌐 GitHub: [KmSasikumar](https://github.com/KmSasikumar)

---

⭐ **If you found this project useful, don't forget to star the repository!**
