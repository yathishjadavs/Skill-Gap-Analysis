#!/bin/bash
set -e
echo "=== Skill Gap Analyzer - Setup ==="

echo "-> Installing Python dependencies..."
cd backend
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo "-> Installing frontend dependencies..."
cd ../frontend
npm install --silent

echo "-> Creating directories..."
mkdir -p ../backend/chroma_data ../backend/uploads

cd ../backend
if [ ! -f .env ]; then
    echo "WARNING: backend/.env not found. Copy .env.example to .env and fill in values."
    exit 1
fi

echo "-> Generating 5000+ courses in MongoDB..."
python scripts/generate_courses.py

echo "-> Running course embedding pipeline (MongoDB -> ChromaDB)..."
python scripts/embed_courses.py

echo ""
echo "Setup complete."
echo "   Backend:  cd backend && uvicorn app.main:app --reload --port 8000"
echo "   Frontend: cd frontend && npm start"
