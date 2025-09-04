from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import asyncio
import os

# Import services and routes
from app.services.stock_data import StockDataService
from app.api.stock_routes import router as stock_router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Stock Signal API")

# Initialize services at startup
@app.on_event("startup")
async def startup_event():
    logger.info("Kicking off background stock data initialization...")
    app.state.stock_data_service = StockDataService()
    asyncio.create_task(app.state.stock_data_service.initialize_async())
    logger.info("Background initialization task started")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a basic test route
@app.get("/test")
async def test_route():
    logger.info("Test route accessed")
    return {"status": "ok"}

# Setup templates with UTF-8 encoding
logger.info("Setting up templates directory: app/templates")
try:
    templates = Jinja2Templates(directory="app/templates")
    logger.info("Templates directory setup successful with UTF-8 encoding")
except Exception as e:
    logger.error(f"Failed to setup templates: {str(e)}", exc_info=True)
    raise

# Basic error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error handling request: {request.url}")
    logger.error(f"Error details: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# Basic health check
@app.get("/health")
async def health_check():
    logger.info("Health check accessed")
    return {"status": "healthy"}

# Test template route
@app.get("/test-template", response_class=HTMLResponse)
async def test_template(request: Request):
    logger.info("Serving test.html")
    return templates.TemplateResponse("test.html", {"request": request})

# Route for serving the main page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        logger.info("Attempting to serve index.html")
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Template directory exists: {os.path.exists('app/templates')}")
        logger.info(f"index.html exists: {os.path.exists('app/templates/index.html')}")
        
        # Try to read the file directly to verify content
        with open('app/templates/index.html', 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"index.html size: {len(content)} bytes")
            logger.info(f"First 100 chars: {content[:100]}")
        
        response = templates.TemplateResponse("index.html", {"request": request})
        # Add no-cache headers
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        logger.info("Successfully rendered index.html with no-cache headers")
        return response
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load UI: {str(e)}"
        )

# Include stock routes
app.include_router(stock_router, prefix="/api/stocks", tags=["stocks"])

if __name__ == "__main__":
    # Using port 8090
    uvicorn.run("app.main:app", host="0.0.0.0", port=8090, reload=True)

# For direct uvicorn command, use:
# python -m uvicorn app.main:app --reload --port 8090