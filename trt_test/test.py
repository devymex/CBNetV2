import numpy as np
from trt_test.trtinfer import TRTInference
model = TRTInference('trt_test/models/backbone.trt')
x = np.zeros((1, 3, 1088, 1600), dtype=np.float32)
model.inference({'x': x})