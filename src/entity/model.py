from typing import Optional, Dict, Any
from sklearn.pipeline import Pipeline
import joblib
import logging

logger = logging.getLogger(__name__)

class Model:
    _current: Optional['Model'] = None
    
    def __init__(self,
                 metadata: Optional[Dict[str, Any]] = None,
                 pipeline: Pipeline = None):
        self._pipeline = pipeline
        self._metadata = metadata

    @property
    def pipeline(self) -> Optional[Pipeline]:
        return self._pipeline
    
    @pipeline.setter
    def pipeline(self, value: Optional[Pipeline]):
        self._pipeline = value

    @property
    def metadata(self) -> Optional[str]:
        return self._metadata
    
    @metadata.setter
    def metadata(self, value: Optional[str]):
        self._metadata = value

    @classmethod
    def get_instance(cls) -> 'Model':
        if cls._current is None:
            cls.load_model()
        return cls._current
    
    @classmethod
    def clear_instance(cls):
        cls._current = None
    
    @classmethod
    def load_model(cls, path: str = './data/model.pkl') -> 'Model':
        """
        Load the model from the given path
        """
        data = joblib.load(path)
        cls._current = Model(pipeline=data['pipeline'], metadata=data['metadata'])
        logging.info("Model loaded")
        
        return cls._current
    
    @classmethod
    def save_model(cls,
                   pipeline: Pipeline,
                   metadata: Dict,
                   path: str = './data/model.pkl') -> None:
        """
        Save the model to the given path
        """
        logging.info(f"Saving model to {path}")
        data = {
            'pipeline': pipeline,
            'metadata': metadata
        }
        joblib.dump(data, path)
        logging.info("Model saved")

        # Update the current instance
        cls._current = Model(pipeline=pipeline, metadata=metadata)
        logging.info("Model instance updated")