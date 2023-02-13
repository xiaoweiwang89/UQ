import tensorflow as tf

from .activation import Dice
from .core import DNN, LocalActivationUnit, PredictionLayer
from .interaction import (CIN, FGCNNLayer)
from .normalization import LayerNormalization
from .utils import NoMask, Hash,Linear,Add

custom_objects = {'tf': tf,
                  'DNN': DNN,
                  'PredictionLayer': PredictionLayer,
                  'LocalActivationUnit': LocalActivationUnit,
                  'Dice': Dice,
                  'CIN': CIN,
                  'LayerNormalization': LayerNormalization,
                  'NoMask': NoMask,
                  'FGCNNLayer': FGCNNLayer,
                  'Hash': Hash,
                  'Linear':Linear,
                  'Add':Add,
                  }
