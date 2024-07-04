from numbers import Number
from typing import Union, Tuple, Dict,List
import torch


from src.utils.helper import save_model

def preprocess_data(model_input: List[torch.Tensor], device):
    return [mi.to(device) for mi in model_input]

def postprocess_loss_results(result: Union[Number, Tuple[Number, Dict[str, Number]]]):
    """
    Utility function that ensures that the loss results have the format of
    (total_loss: float-like, sub_losses: dict[float-like])
    so that we can do easier processing (and logging) of results.
    """
    # we also want to allow regular loss functions that do not return additional partial losses
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result
    elif len(result) == 1:
        return result, {}
    else:
        raise SystemError("This type of loss result is not yet supported.")


def add_to_dict(original: dict, new: dict, multiplier: float = 1.) -> dict:
    for k, l in new.items():
        original[k] += l * multiplier
    return original


def scale_dict_items(d: dict, scale: float) -> dict:
    return {k: v * scale for k, v in d.items()}


def process_results(model, epoch, early_stopping_criteria, results_dict, best_validation_scores,
                    result_path, is_verbose=False):
    for label, criterion in early_stopping_criteria.items():
        if (wu := criterion.get("warmup")) and epoch <= wu:
            break

        # Allow early stopping on non rank-based metrics
        validation_score = results_dict[criterion["metric"]]
        if "top_k" in criterion:
            validation_score = validation_score[criterion["top_k"]]

        better_model_found = False
        if "highest_is_best" in criterion:
            better_model_found = validation_score >= best_validation_scores[label] and criterion[
                "highest_is_best"] or validation_score <= best_validation_scores[label] and not criterion[
                "highest_is_best"]
        elif "closest_is_best" in criterion:
            old_diff = abs(criterion["value"] - best_validation_scores[label])
            new_diff = abs(criterion["value"] - validation_score)
            better_model_found = new_diff <= old_diff

        if better_model_found:
            if is_verbose:
                print("Better model found!")
                if top_k := criterion.get("top_k"):
                    print(f'{criterion["metric"]}@{top_k}={validation_score:.4f}\n')
                else:
                    print(f'{criterion["metric"]}={validation_score:.4f}\n')
            save_model(model, result_path, "best_model_" + label)
            best_validation_scores[label] = validation_score

    return best_validation_scores
