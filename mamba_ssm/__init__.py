__version__ = "2.2.4"

# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
# from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.modules.mamba2 import Mamba2
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
try:
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError as e:
    print(f"Error importing Mamba2: {e}")
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

__all__ = ["Mamba", "MambaLMHeadModel"]  # Dùng Mamba nếu Mamba2 thất bại