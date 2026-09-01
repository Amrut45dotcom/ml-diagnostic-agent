EXPERIMENT_BANK = {
    "try_lower_lr_001": {
        "description": "Retrain with LR=0.01 and observe the result",
        "summary": (
            "LR=0.01: train_loss 0.181 -> 0.055, val_loss 0.099 -> 0.054, "
            "grad_norm 0.808 -> 0.403, test_accuracy 98.53%"
        ),
    },
    "try_alternate_high_lr_01": {
        "description": "Retrain with LR=0.1 and observe the result",
        "summary": (
            "LR=0.1: train_loss 2.378 -> 2.309, val_loss 2.314 -> 2.321, "
            "grad_norm 0.229 -> 0.120, test_accuracy 9.58% "
            "(loss stuck near ln(10)=2.3, model not learning — random-guess baseline)"
        ),
    },
}