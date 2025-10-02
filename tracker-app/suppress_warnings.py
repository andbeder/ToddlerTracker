"""
Suppress common harmless warnings from PyTorch and other libraries.
Import this at the top of app.py to clean up logs.
"""

import warnings
import os

# Suppress PyTorch CUDA DSA warnings (informational only)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Suppress torchreid Cython warning
warnings.filterwarnings('ignore', message='Cython evaluation.*is unavailable')

# Suppress other common warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchreid')
