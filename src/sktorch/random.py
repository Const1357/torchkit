def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility
    """
    import torch
    import random
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    random.seed(seed)
    np.random.seed(seed)