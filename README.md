# Client Action Prediction Project

## Description

This project demonstrates essential Data Science and Machine Learning skills. The primary goal is to create a model that predicts whether a client will take a target action on a website (e.g., make a purchase, register, etc.) based on client characteristics such as acquisition method, device, operating system, browser, and location. The model was trained on a large dataset and deployed as a web service using FastAPI.

## Structure and Technologies Used

### Jupyter Notebook

- **Purpose**: Data analysis and preparation for model creation.
- **Description**: In the initial project stage, the data was loaded and explored in Jupyter Notebook, enabling data analysis and visualization and assessing data quality. These preliminary steps help prepare the data for machine learning to improve the model's prediction quality.

### Machine Learning Model

- **Methodology**: The model was built as a **pipeline**.
  - **Pipeline** — organizes the data processing and prediction steps sequentially (e.g., data normalization, classification).
- **Model Saving**: The model was saved in `.pkl` format using the `dill` library, allowing for the storage of complex Python objects.
  - **.pkl** — a binary format for serializing objects, enabling the model to be saved to disk and loaded as needed.

### FastAPI

- **Purpose**: Deploy the model as an API service.
- **Description**: FastAPI is a high-performance web framework used to create APIs. FastAPI enabled the creation of a web service that accepts input data, processes it, and returns a prediction result.

## Details of `main.py`

The `main.py` file contains the core code for the service:

1. **Import Required Libraries**:
    - `dill` — used to load the pre-saved model from the `.pkl` file.
    - `pandas` — to convert input data into a format compatible with the model.
    - `fastapi` and `pydantic` — to create the API service and validate input data.

2. **Initializing FastAPI**:
    - `app = FastAPI()` — creates a FastAPI application instance to handle requests.

3. **Loading the Model**:
    - The model is loaded from the `model.pkl` file and saved in the `model` variable, allowing it to be used for predictions upon receiving requests.

4. **Defining Input Data**:
    - The `Form` class, based on `BaseModel` from `pydantic`, defines the parameters required for prediction (e.g., `utm_source`, `device_category`, `geo_country`, etc.).
    - This class is used to validate input data when sending requests to the API.

5. **Defining Output Data**:
    - The `Prediction` class represents the structure of the API response. It contains a `prediction` field that returns the prediction result.

6. **API Endpoints**:
    - `/status`: Returns "I'm OK" to check if the service is running.
    - `/version`: Returns metadata about the model, such as version information.
    - `/predict`: The main route for obtaining a prediction. Accepts data in the `Form` format, converts it into a DataFrame, and uses the model to generate a prediction. The result is returned in `Prediction` format.

## How to Run the Project

1. **Install Dependencies**:
    ```bash
    pip install fastapi dill pandas pydantic
    ```

2. **Run the FastAPI Service**:
    ```bash
    uvicorn main:app --reload
    ```
   - `uvicorn` — an ASGI server to run applications written with FastAPI.
   - The `--reload` flag enables automatic reloading of the app when code changes.

3. **Testing the Service**:
   - Once running, the service is accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000).
   - You can check the service status by sending a GET request to `/status`.
   - To obtain a prediction, send a POST request to `/predict`, passing a JSON object with the parameters defined in the `Form` class.

## Conclusion

This project demonstrates how Python and various libraries can be used to create a complete machine learning service. It includes all stages, from initial data analysis to building and deploying the model as an API, allowing for the predictive model to be integrated into a real application.
