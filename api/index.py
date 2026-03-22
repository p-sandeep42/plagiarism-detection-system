from fastapi import FastAPI
import sys
import os

# Append the api directory to the path so local imports inside api/ work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now we can safely import our app from main.py
from main import app as original_app

# Vercel Serverless Functions look for the `app` variable in api/index.py
# If the frontend calls /api/..., we need to mount the original_app to `/api` or matching paths.
# Since our Next.js frontend is configured to call `/api/py/...`, we mount it there.
app = FastAPI()

app.mount("/api/py", original_app)
