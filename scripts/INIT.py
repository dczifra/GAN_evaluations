
sample_size=2000
# ===== Generate mnist train/test datasets =====
from process_model import Models
import os
Models.generate_mnist(sample_size,True)
Models.generate_mnist(sample_size,False)
