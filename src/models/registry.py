"""Model registry mapping names to constructors and train functions."""
from typing import Tuple, Callable, Any, Dict

from .random_forest_model import create_rf_model, train_model as train_rf
from .xgboost_model import create_xgb_model, train_model as train_xgb
from .svm_model import create_svm_model, train_model as train_svm
from .mlp_model import create_mlp_model, train_model as train_mlp
from config.model_config import MODEL_PARAMS

# Mapping of short model keys to (create_function, train_function)
MODEL_REGISTRY: Dict[str, Tuple[Callable[..., Any], Callable[..., Any]]] = {
    "rf": (create_rf_model, train_rf),
    "xgb": (create_xgb_model, train_xgb),
    "svm": (create_svm_model, train_svm),
    "mlp": (create_mlp_model, train_mlp),
}


def get_model(name: str, **kwargs) -> Tuple[Any, Callable[..., Any]]:
    """Instantiate a model by name and return it with its training function.

    Parameters
    ----------
    name: str
        Registry key of the model to create.
    **kwargs: dict
        Additional keyword arguments to override default hyperparameters.

    Returns
    -------
    tuple
        (model_instance, train_function)
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found in registry")

    create_func, train_func = MODEL_REGISTRY[name]
    params = MODEL_PARAMS.get(name, {}).copy()
    params.update(kwargs)
    model = create_func(params)
    return model, train_func
