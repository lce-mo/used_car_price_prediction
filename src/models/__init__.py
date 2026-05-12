"""Model training, prediction, evaluation, cross-validation, and registry helpers."""

__all__ = [
    "cross_validate_train",
    "evaluate_regression",
    "fit_full_and_predict",
    "fit_predict_model",
    "get_model",
    "list_models",
    "save_outputs",
]


def __getattr__(name: str):
    if name == "cross_validate_train":
        from src.models.cross_validation import cross_validate_train

        return cross_validate_train
    if name == "evaluate_regression":
        from src.models.evaluate_model import evaluate_regression

        return evaluate_regression
    if name in {"get_model", "list_models", "save_outputs"}:
        from src.models import model_registry

        return getattr(model_registry, name)
    if name == "fit_full_and_predict":
        from src.models.predict_model import fit_full_and_predict

        return fit_full_and_predict
    if name == "fit_predict_model":
        from src.models.train_model import fit_predict_model

        return fit_predict_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
