# LifeSignal Backend

This is the backend service for the LifeSignal application, which provides health data analysis using MongoDB and Gemini AI.

## Setup

1. Make sure you have Python 3.8+ installed
2. Create a `.env` file with the required environment variables (see `.env.example`)
3. Run `./run.sh` to start the server

## Environment Variables

The following environment variables should be in your `.env` file:

```
# LifeSignal Backend Environment Variables

GEMINI_API_KEY=your_gemini_api_key
MONGODB_URI=your_mongodb_uri
JWT_SECRET=your_jwt_secret

FLASK_APP=app.py
FLASK_ENV=development
# Set to 1 to enable debug logs
DEBUG=1
```

## Project Structure

```
backend/
├── app.py                 # Main application file
├── config.py              # Configuration loader
├── database.py            # MongoDB connection handler
├── gemini_client.py       # Gemini AI client
├── utils.py               # Utility functions
├── models/                # Database models
│   ├── health_data.py     # Health data model
│   └── user.py            # User model
├── routes/                # API routes
│   ├── auth_routes.py     # Authentication routes
│   └── health_routes.py   # Health data routes
├── services/              # Business logic services
│   ├── auth_service.py    # Authentication service
│   └── health_service.py  # Health data analysis service
├── requirements.txt       # Python dependencies
└── run.sh                 # Startup script
```

## API Endpoints

### Authentication

- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login and get JWT token

### Health Data

- `POST /api/health/analyze` - Analyze health data (requires authentication)
- `GET /api/health/history` - Get user's health data history (requires authentication)

### Miscellaneous

- `GET /health` - Health check endpoint

## Running the Application

```bash
./run.sh
```

This will start the server on http://0.0.0.0:5100 by default.
