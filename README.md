# Smart Document Verification System Core

An advanced AI-powered system for automatically verifying, classifying, and extracting key information from Sri Lankan national identity card (new and old), driving licence and passport.  

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-000000)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Document Classification**: Automatically identifies the type of uploaded document.
  - New NIC (National Identity Card)
  - Old NIC
  - Driving License
  - Passport
- **OCR Data Extraction**: Precisely extracts crucial text fields like NIC numbers, driving license numbers, and passport numbers.
- **RESTful API**: Simple and robust HTTP API for easy integration into any application.
- **Optimized for Production**: Utilizes TensorFlow Lite for fast inference on both server and edge devices.

## System Architecture

The system is built on a multi-stage pipeline:

1.  **Image Preprocessing**: Standardizes input images for optimal model performance.
2.  **Document Classification**: A custom-trained Convolutional Neural Network (CNN) classifies the document into one of four categories.
3.  **OCR Processing**: For identified IDs, a Convolutional Recurrent Neural Network (CRNN) extracts specific text fields.
4.  **API Response**: Results are structured into a JSON response for easy consumption.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/arvinjayanake/smart-doc-verification.git
    cd smart-doc-verification
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the API Server

1.  **Start the Flask development server**:
    ```bash
    python app.py
    ```
    *The API will start running on `http://127.0.0.1:5000`.*

2.  **Verify the server is running**:
    Open your browser or use `curl`:
    ```bash
    curl http://127.0.0.1:5000/
    ```
    You should receive a welcome message.

## 📡 API Usage

### Endpoint: `POST /api/verify_image`

Analyzes an uploaded image, classifies the document type, and extracts relevant information.

#### Request

- **URL**: `http://127.0.0.1:5000/api/verify_image`
- **Method**: `POST`
- **Headers**:
  - `Content-Type: application/json`
- **Body** (JSON):
  ```json
  {
      "img_data": "<base64_encoded_image_string>"
  }