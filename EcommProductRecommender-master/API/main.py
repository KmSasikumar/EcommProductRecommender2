# main.py
import os
import json
import shutil
import uuid
import pandas as pd
import numpy as np
import sqlite3
from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Depends, Body
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from enum import Enum
import tensorflow as tf
from tensorflow.keras.models import load_model
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Assuming train.py and model.py are in the same src directory
from train import load_and_preprocess_data, train_model, save_mappings, load_mappings

# --- Configuration & Globals ---
API_KEY_NAME = "X-API-Key"
MODELS_BASE_DIR = "models_store"
INTERACTIONS_DB_PATH = "user_interactions.db"
PRODUCTS_DB_PATH = "ecommerce.db"  # Path for the SQLite products database file

# Fix the API_KEYS_DB model path that was cut off
API_KEYS_DB = {
    "testkey123": {
        "model_path": os.path.join(MODELS_BASE_DIR, "testkey123", "ncf_model.h5"),
        "mappings_path": os.path.join(MODELS_BASE_DIR, "testkey123", "ncf_mappings.json"),
    }
}

os.makedirs(MODELS_BASE_DIR, exist_ok=True)

# --- Database Initialization Function ---
def init_databases():
    """
    Creates the user_interactions and products tables in the SQLite databases if they don't exist.
    """
    # Initialize interactions DB
    conn_int = sqlite3.connect(INTERACTIONS_DB_PATH)
    c_int = conn_int.cursor()
    c_int.execute('''
        CREATE TABLE IF NOT EXISTS user_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            type TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    ''')
    conn_int.commit()
    conn_int.close()
    print(f"Interactions database initialized at {INTERACTIONS_DB_PATH}")

    # Initialize products DB
    conn_prod = sqlite3.connect(PRODUCTS_DB_PATH)
    c_prod = conn_prod.cursor()
    c_prod.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT,
            image_urls TEXT,
            tags TEXT
        )
    ''')
    conn_prod.commit()
    conn_prod.close()
    print(f"Products database initialized at {PRODUCTS_DB_PATH}")

init_databases()

# Create FastAPI app
app = FastAPI(title="E-commerce Recommendation System API", version="0.1.0")

# ADD CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Key Authentication ---
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(key: str = Security(api_key_header)):
    if key in API_KEYS_DB:
        return key
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# --- Pydantic Models for Request/Response ---
class TrainResponse(BaseModel):
    message: str
    api_key: str
    model_path: str
    mappings_path: str

class RecommendationRequest(BaseModel):
    user_id: str
    count: int = 10
    search_query: str = ""

class RecommendationItem(BaseModel):
    item_id: str
    score: float

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    user_id: str

class InteractionType(str, Enum):
    tap = "tap"
    cart = "cart"

class InteractionRequest(BaseModel):
    user_id: str
    item_id: str
    type: InteractionType
    timestamp: Optional[float] = None

class InteractionResponse(BaseModel):
    message: str
    success: bool

class Product(BaseModel):
    id: str
    name: str
    price: float
    imageUrls: List[str]
    category: str
    tags: List[str]

class SearchResponse(BaseModel):
    products: List[Product]

# --- Helper Functions ---
def get_model_and_mappings_for_key(api_key: str):
    if api_key not in API_KEYS_DB or \
       not os.path.exists(API_KEYS_DB[api_key].get("model_path", "")) or \
       not os.path.exists(API_KEYS_DB[api_key].get("mappings_path", "")):
        raise HTTPException(status_code=404, detail="Model or mappings not found for this API key. Ensure the model is trained or paths are correct.")

    model_data = API_KEYS_DB[api_key]
    try:
        if not os.path.exists(model_data["model_path"]):
            raise HTTPException(status_code=404, detail=f"Model file not found at {model_data['model_path']}")
        if not os.path.exists(model_data["mappings_path"]):
            raise HTTPException(status_code=404, detail=f"Mappings file not found at {model_data['mappings_path']}")

        model = load_model(model_data["model_path"])
        user_map, item_map = load_mappings(model_data["mappings_path"])
        idx_to_item_map = {idx: item_id for item_id, idx in item_map.items()}
        return model, user_map, item_map, idx_to_item_map, model_data.get("num_users"), model_data.get("num_items")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/mappings: {str(e)}")

# --- NEW: Helper function to search products in the database ---
def search_products_in_db(query_term: str):
    """
    Searches for products in the SQLite database based on name, category, or tags.
    Args:
        query_term: The search string provided by the user.
                   If empty, returns all products.
    Returns:
        A list of Product Pydantic objects representing the matching products.
    """
    if not query_term:
        # Return all products if no query
        conn_all = None
        try:
            conn_all = sqlite3.connect(PRODUCTS_DB_PATH)
            conn_all.row_factory = sqlite3.Row
            c_all = conn_all.cursor()
            c_all.execute('''
                SELECT id, name, price, category, image_urls, tags
                FROM products
            ''')
            rows_all = c_all.fetchall()
            all_products = []
            for row in rows_all:
                product_id = row['id']
                name = row['name']
                price = row['price']
                category = row['category']
                image_urls_json = row['image_urls']
                tags_json = row['tags']
                try:
                    image_urls = json.loads(image_urls_json) if image_urls_json else []
                    tags = json.loads(tags_json) if tags_json else []
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON for product {product_id}. Using empty lists.")
                    image_urls = []
                    tags = []
                all_products.append(Product(
                    id=product_id,
                    name=name,
                    price=price,
                    category=category,
                    imageUrls=image_urls,
                    tags=tags
                ))
            return all_products
        except sqlite3.Error as e:
            print(f"Database error fetching all products: {e}")
            return []
        finally:
            if conn_all:
                conn_all.close()
        return []

    conn = None
    try:
        conn = sqlite3.connect(PRODUCTS_DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        search_pattern = f"%{query_term.lower()}%"
        c.execute('''
            SELECT id, name, price, category, image_urls, tags
            FROM products
            WHERE LOWER(name) LIKE ?
               OR LOWER(category) LIKE ?
               OR LOWER(tags) LIKE ?
        ''', (search_pattern, search_pattern, search_pattern))

        rows = c.fetchall()

        search_results = []
        for row in rows:
            product_id = row['id']
            name = row['name']
            price = row['price']
            category = row['category']
            image_urls_json = row['image_urls']
            tags_json = row['tags']

            try:
                image_urls = json.loads(image_urls_json) if image_urls_json else []
                tags = json.loads(tags_json) if tags_json else []
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON for product {product_id}. Using empty lists.")
                image_urls = []
                tags = []

            search_results.append(Product(
                id=product_id,
                name=name,
                price=price,
                category=category,
                imageUrls=image_urls,
                tags=tags
            ))

        return search_results

    except sqlite3.Error as e:
        print(f"Database error during search: {e}")
        return []
    finally:
        if conn:
            conn.close()

# --- API Endpoints ---
@app.post("/v1/train", response_model=TrainResponse)
async def train_new_model(training_data: UploadFile = File(...)):
    temp_file_path = None
    new_api_key_generated = None
    try:
        new_api_key = str(uuid.uuid4())
        new_api_key_generated = new_api_key

        model_dir = os.path.join(MODELS_BASE_DIR, new_api_key)
        os.makedirs(model_dir, exist_ok=True)

        model_save_path = os.path.join(model_dir, "ncf_model.h5")
        mappings_save_path = os.path.join(model_dir, "ncf_mappings.json")

        temp_file_path = f"temp_{new_api_key}_{training_data.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(training_data.file, buffer)

        print(f"Training data saved to {temp_file_path}")

        df_processed, user_map, item_map, num_users, num_items = load_and_preprocess_data(temp_file_path)

        if df_processed.empty or num_users == 0 or num_items == 0:
            raise HTTPException(status_code=400, detail="Processed data is empty or no users/items found. Check data format and content.")

        trained_model, training_history = train_model(
            df_processed, num_users, num_items,
            model_save_path=model_save_path
        )

        save_mappings(user_map, item_map, mappings_save_path)

        API_KEYS_DB[new_api_key] = {
            "model_path": model_save_path,
            "mappings_path": mappings_save_path,
            "num_users": num_users,
            "num_items": num_items
        }

        return TrainResponse(
            message="Model training initiated and completed successfully.",
            api_key=new_api_key,
            model_path=model_save_path,
            mappings_path=mappings_save_path
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        if new_api_key_generated and os.path.exists(os.path.join(MODELS_BASE_DIR, new_api_key_generated)):
             shutil.rmtree(os.path.join(MODELS_BASE_DIR, new_api_key_generated), ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during training: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if training_data:
            await training_data.close()


# Replace the /recommend endpoint in your main.py with this:

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations_legacy(
    request: RecommendationRequest,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Hybrid recommendation endpoint:
    1. Generates NCF recommendations based on user behavior
    2. If search_query provided, returns all DB-matching products ordered by NCF score
    3. Falls back to top-N NCF recommendations if DB search returns nothing
    """
    model, user_map, item_map, idx_to_item_map, num_users, num_items = get_model_and_mappings_for_key(api_key)

    if request.user_id not in user_map:
        raise HTTPException(status_code=404, detail=f"User ID '{request.user_id}' not found in the model's user mapping.")

    user_idx = user_map[request.user_id]
    all_item_indices = np.array(list(item_map.values()))

    if num_users is None or num_items == 0:
         raise HTTPException(status_code=500, detail="Number of items not available for model, cannot generate candidates.")

    candidate_item_indices = all_item_indices

    if candidate_item_indices.size == 0:
         raise HTTPException(status_code=404, detail="No candidate items found for recommendation.")

    user_array = np.full(len(candidate_item_indices), user_idx)
    predictions = model.predict([user_array, candidate_item_indices], batch_size=512)

    # Generate NCF recommendations (scores for all candidates)
    results = []
    for i, item_idx_val in enumerate(candidate_item_indices):
        original_item_id = idx_to_item_map.get(item_idx_val)
        if original_item_id:
             results.append({"item_id": original_item_id, "score": float(predictions[i][0])})

    # Keep a global results list sorted by score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)
    top_n_recommendations = results[:request.count]

    # HYBRID LOGIC: If search_query provided, return all DB-matching products,
    # ordered by their NCF score (score 0.0 if the model didn't score that product).
    if request.search_query and request.search_query.strip():
        print(f"Search query provided: '{request.search_query}'. Applying hybrid ordering (DB matches ordered by NCF score)...")
        db_search_results = search_products_in_db(request.search_query)
        print(f"DB search returned {len(db_search_results)} products")

        if db_search_results:
            # Build a score map from all NCF results
            score_map = {rec["item_id"]: rec["score"] for rec in results}
            # Combine DB search results with NCF scores (default 0.0 when missing)
            combined = []
            for idx, p in enumerate(db_search_results):
                s = float(score_map.get(p.id, 0.0))
                combined.append({"item_id": p.id, "score": s, "db_index": idx})

            # Sort by score desc, tiebreaker by original DB order (db_index) to preserve DB relevance for ties
            combined.sort(key=lambda x: (-x["score"], x["db_index"]))
            # Use top 'count' from combined list
            top_n_recommendations = [{"item_id": c["item_id"], "score": c["score"]} for c in combined[: request.count]]
            print(f"Returning {len(top_n_recommendations)} DB-matching products ordered by NCF score.")
        else:
            # DB returned nothing -> fallback to top-N NCF recommendations
            print("DB search returned no matching products; falling back to top-N NCF recommendations.")
            top_n_recommendations = results[: request.count]
    else:
        print("No search query provided. Returning top-N NCF recommendations.")

    return RecommendationResponse(
        recommendations=[RecommendationItem(**item) for item in top_n_recommendations],
        user_id=request.user_id
    )


# Also update /v1/recommendations endpoint with the same logic:

@app.post("/v1/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Hybrid recommendation endpoint (v1):
    Generates item recommendations for a given user_id with optional search filtering.
    """
    try:
        model, user_map, item_map, idx_to_item_map, num_users, num_items = get_model_and_mappings_for_key(api_key)

        if request.user_id not in user_map:
            raise HTTPException(status_code=404, detail=f"User ID '{request.user_id}' not found in the model's user mapping.")

        user_idx = user_map[request.user_id]
        all_item_indices = np.array(list(item_map.values()))

        if num_items is None or num_items == 0:
             raise HTTPException(status_code=500, detail="Number of items not available for model, cannot generate candidates.")

        candidate_item_indices = all_item_indices

        if candidate_item_indices.size == 0:
             raise HTTPException(status_code=404, detail="No candidate items found for recommendation.")

        user_array = np.full(len(candidate_item_indices), user_idx)
        predictions = model.predict([user_array, candidate_item_indices], batch_size=512)

        # Generate NCF recommendations (scores for all candidates)
        results = []
        for i, item_idx_val in enumerate(candidate_item_indices):
            original_item_id = idx_to_item_map.get(item_idx_val)
            if original_item_id:
                 results.append({"item_id": original_item_id, "score": float(predictions[i][0])})

        results.sort(key=lambda x: x["score"], reverse=True)
        top_n_recommendations = results[:request.count]

        # HYBRID LOGIC: If search_query provided, return all DB-matching products,
        # ordered by their NCF score (score 0.0 if the model didn't score that product).
        if request.search_query and request.search_query.strip():
            print(f"Search query provided: '{request.search_query}'. Applying hybrid ordering (DB matches ordered by NCF score)...")
            db_search_results = search_products_in_db(request.search_query)
            print(f"DB search returned {len(db_search_results)} products")

            if db_search_results:
                score_map = {rec["item_id"]: rec["score"] for rec in results}
                combined = []
                for idx, p in enumerate(db_search_results):
                    s = float(score_map.get(p.id, 0.0))
                    combined.append({"item_id": p.id, "score": s, "db_index": idx})

                combined.sort(key=lambda x: (-x["score"], x["db_index"]))
                top_n_recommendations = [{"item_id": c["item_id"], "score": c["score"]} for c in combined[: request.count]]
                print(f"Returning {len(top_n_recommendations)} DB-matching products ordered by NCF score.")
            else:
                print("DB search returned no matching products; falling back to top-N NCF recommendations.")
                top_n_recommendations = results[: request.count]
        else:
            print("No search query provided. Returning top-N NCF recommendations.")

        return RecommendationResponse(
            recommendations=[RecommendationItem(**item) for item in top_n_recommendations],
            user_id=request.user_id
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"An unexpected error occurred during recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during recommendation: {str(e)}")

# --- Search Endpoint ---
@app.post("/search", response_model=SearchResponse)
async def search_products_endpoint(
    query: str = Body(..., embed=True)
):
    """
    Searches for products based on name, category, or tags.
    Queries the SQLite database (ecommerce.db) for results.
    This is separate from the NCF recommendation logic.
    """
    print(f"Received search query: '{query}'")

    search_results_list = search_products_in_db(query)

    print(f"Returning {len(search_results_list)} search results.")
    return SearchResponse(products=search_results_list)

# --- Interaction Tracking Endpoint ---
@app.post("/interactions", response_model=InteractionResponse)
async def log_interaction(
    interaction: InteractionRequest
):
    interaction_timestamp = interaction.timestamp or time.time()

    try:
        conn = sqlite3.connect(INTERACTIONS_DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO user_interactions (user_id, item_id, type, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (interaction.user_id, interaction.item_id, interaction.type, interaction_timestamp))
        conn.commit()
        conn.close()
        print(f"Interaction stored in DB: User '{interaction.user_id}' performed '{interaction.type}' on item '{interaction.item_id}' at {interaction_timestamp}")
    except sqlite3.Error as e:
        print(f"Database error storing interaction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store interaction in database: {str(e)}")

    return InteractionResponse(message="Interaction logged successfully", success=True)

# --- Retrain Model Endpoint ---
executor = ThreadPoolExecutor(max_workers=1)

from retrain_model import retrain_model_with_new_data

@app.post("/retrain", dependencies=[Depends(get_api_key)])
async def trigger_retrain():
    print("Retrain endpoint called. Initiating retraining in background thread...")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, retrain_model_with_new_data)

        print("Retraining process initiated/completed in background thread.")
        return {"message": "Model retraining process initiated successfully.", "status": "started"}
    except Exception as e:
        print(f"Error initiating retraining process: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start retraining: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)
    print("ThreadPoolExecutor for retraining shut down.")

@app.get("/")
async def root():
    return {"message": "Welcome to the E-commerce Recommendation System API. See /docs for API details."}

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server with Uvicorn...")

    # Setup for 'testkey123' for local development demonstration
    if "testkey123" in API_KEYS_DB:
        _model_info = API_KEYS_DB["testkey123"]
        _dummy_model_dir = os.path.dirname(_model_info["model_path"])
        os.makedirs(_dummy_model_dir, exist_ok=True)
        _dummy_model_path = _model_info["model_path"]
        _dummy_mappings_path = _model_info["mappings_path"]

        # Check if either model or mappings are missing to create dummies
        if not os.path.exists(_dummy_model_path) or not os.path.exists(_dummy_mappings_path):
            print(f"Creating dummy model/mappings for 'testkey123' in {_dummy_model_dir}...")
            try:
                dummy_users_count = 10
                dummy_items_count = 5

                # Create a placeholder for the model file if it doesn't exist
                if not os.path.exists(_dummy_model_path):
                    # For TensorFlow/Keras, a .h5 file is not just text.
                    # A minimal valid HDF5 file is complex to create from scratch.
                    # So, we'll create a dummy model using Keras and save it.
                    # This requires model.py to be importable.
                    try:
                        from .model import create_ncf_model # Relative import
                        temp_model = create_ncf_model(dummy_users_count, dummy_items_count, embedding_dim=4) # minimal
                        temp_model.save(_dummy_model_path)
                        print(f"Dummy Keras model saved to {_dummy_model_path}")
                    except Exception as model_ex:
                        print(f"Could not create full dummy Keras model due to: {model_ex}. Creating placeholder file.")
                        with open(_dummy_model_path, 'w') as f: f.write("dummy model placeholder - not a valid Keras model")


                # Create dummy mappings if they don't exist
                if not os.path.exists(_dummy_mappings_path):
                    dummy_user_map = {f"user{i}": i for i in range(dummy_users_count)}
                    dummy_item_map = {f"item{j}": j for j in range(dummy_items_count)}
                    save_mappings(dummy_user_map, dummy_item_map, _dummy_mappings_path)
                    print(f"Dummy mappings saved to {_dummy_mappings_path}")

                # Update API_KEYS_DB with counts for the dummy model
                API_KEYS_DB["testkey123"]["num_users"] = dummy_users_count
                API_KEYS_DB["testkey123"]["num_items"] = dummy_items_count
                print(f"Dummy model and mappings for 'testkey123' setup in {_dummy_model_dir}.")
            except Exception as e:
                print(f"Error creating dummy model/mappings for testkey123: {e}")
        else:
            # If files exist, ensure num_users/num_items are in API_KEYS_DB for testkey123
            if "num_users" not in API_KEYS_DB["testkey123"] or "num_items" not in API_KEYS_DB["testkey123"]:
                try:
                    _, item_map_loaded = load_mappings(_dummy_mappings_path) # Corrected to get both maps
                    user_map_loaded, _ = load_mappings(_dummy_mappings_path) # Load user_map as well
                    API_KEYS_DB["testkey123"]["num_users"] = len(user_map_loaded)
                    API_KEYS_DB["testkey123"]["num_items"] = len(item_map_loaded)
                    print(f"Loaded counts for existing 'testkey123' model: {len(user_map_loaded)} users, {len(item_map_loaded)} items.")
                except Exception as e:
                    print(f"Could not load num_users/num_items for existing testkey123: {e}")
            else:
                 print(f"Dummy model/mappings for 'testkey123' already exist and configured at {_dummy_model_dir}.") 


    uvicorn.run(app, host="0.0.0.0", port=8000)
