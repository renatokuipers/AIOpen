from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field, PositiveInt, confloat


class TrainingConfig(BaseModel):
    learning_rate: confloat(gt=0, le=10) = Field(0.001, description="Learning rate")
    batch_size: PositiveInt = Field(32, description="Batch size")
    optimizer_type: Literal["adam", "sgd", "rmsprop"] = Field(
        "adam", description="Optimizer type"
    )
    hidden_layer_sizes: List[PositiveInt] = Field([64], description="Hidden layer sizes")
    activation_functions: List[Literal["relu", "tanh", "sigmoid"]] = Field(
        ["relu"], description="Activation functions for each hidden layer"
    )
    dropout_rate: confloat(ge=0, le=1) = Field(0.0, description="Dropout rate")
    weight_decay: confloat(ge=0, le=1) = Field(0.0, description="Weight decay")
    epochs: PositiveInt = Field(3, description="Number of training epochs")

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

