import os
import logging
import secrets
from typing import Annotated, Optional
from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    Security,
    Depends
)
from fastapi.background import BackgroundTasks
from fastapi.responses import RedirectResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.status import (
    HTTP_200_OK,
    HTTP_403_FORBIDDEN,
    HTTP_412_PRECONDITION_FAILED,
    HTTP_500_INTERNAL_SERVER_ERROR)
from starlette.responses import JSONResponse
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.repository.common import get_session
from src.service.model import (
    train_model_from_scratch,
    predict,
    all_algorithms,
)
from src.entity.model import Model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------

load_dotenv()
FASTAPI_API_KEY = os.getenv("FASTAPI_API_KEY")
safe_clients = ['127.0.0.1']

api_key_header = APIKeyHeader(name='Authorization', auto_error=False)

async def validate_api_key(request: Request, key: str = Security(api_key_header)):
    '''
    Check if the API key is valid

    Args:
        key (str): The API key to check
    
    Raises:
        HTTPException: If the API key is invalid
    '''
    if request.client.host not in safe_clients and not secrets.compare_digest(str(key), str(FASTAPI_API_KEY)):
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Unauthorized - API Key is wrong"
        )
    return None

app = FastAPI(dependencies=[Depends(validate_api_key)] if FASTAPI_API_KEY else None,
              title="Fraud detection ML API")


# ------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def redirect_to_docs():
    '''
    Redirect to the API documentation.
    '''
    return RedirectResponse(url='/docs')


# ------------------------------------------------------------------------------
@app.get("/train_model", tags=["model"])
async def train_model(background_tasks: BackgroundTasks,
                      limit: Optional[int] = None,
                      algorithm: Optional[all_algorithms] = 'MLP'):
    """
    Train the model
    """
    background_tasks.add_task(
        func=train_model_from_scratch,
        limit=limit,
        evaluate=False,
        algo=algorithm)
    
    return {"message": "Model training in progress"}

class ModelInput(BaseModel):
    transaction_category: str = Field(description='The category of product of the transaction.', example='personal_care')
    transaction_amount: float = Field(gt=0, description="The amount of the transaction", example=2.86)
    customer_job: str = Field(description='The job of the customer.', example='Mechanical engineer')
    customer_address_state: str = Field(description='The state of the customer.', example='SC')
    customer_address_city: str = Field(description='The city of the customer.', example='Columbia')
    customer_address_city_population: int = Field(gt=0, description="The population of the city", example=100000)

class ModelOutput(BaseModel):
    result: int = Field(description="The prediction result. 1 if the transaction is fraudulent, 0 otherwise.", example=1)
    fraud_probability: float = Field(description="The probability of the transaction being fraudulent.", example=0.95)
    model_metadata: dict = Field(description="The metadata of the model.", example={"model_name": "MLP", "version": "1.0"})

@app.post("/predict",
         tags=["model"],
         description="Predict the fraudulent transactions",
         response_description="Prediction result",
         response_model=ModelOutput)
async def make_prediction(params: ModelInput):
    """
    Predict the fraudulent nature of a transaction
    """
    # check the presence of 'model.pkl' file in data/
    if not os.path.exists("./data/model.pkl"):
        raise HTTPException(
            status_code=HTTP_412_PRECONDITION_FAILED, detail="Model not trained. Please train the model first.")

    # Load the model
    model = Model.get_instance()

    # Make the prediction
    prediction = predict(
        pipeline=model.pipeline,
        job=params.customer_job,
        city=params.customer_address_city,
        state=params.customer_address_state,
        category=params.transaction_category,
        amt=params.transaction_amount,
        city_pop=params.customer_address_city_population
    )

    logging.info(prediction)

    # Return the prediction
    return {
        "result": prediction['result'],
        "fraud_probability": prediction['fraud_probability'],
        "model_metadata": model.metadata
    }

# ------------------------------------------------------------------------------
@app.get("/check_health", tags=["general"], description="Check the health of the API")
async def check_health(session: Annotated[Session, Depends(get_session)]):
    """
    Check all the services in the infrastructure are working
    """
    try:
        session.execute(text("SELECT 1"))
        return JSONResponse(content={"status": "healthy"}, status_code=HTTP_200_OK)
    except Exception as e:
        logging.error(f"DB check failed: {e}")
        return JSONResponse(content={"status": "unhealthy"}, status_code=HTTP_500_INTERNAL_SERVER_ERROR)
