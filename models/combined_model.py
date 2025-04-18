from models.st_npp import STNPP
from models.qal import QAL
from models.task_networks.detector import TaskNetwork
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stnpp = STNPP()
        self.qal = QAL()
        self.task_head = TaskNetwork()  # Default: object detection

    def forward(self, x, qp):
        features = self.stnpp(x)
        modulated = self.qal(features, qp)
        output = self.task_head(modulated)
        return output
