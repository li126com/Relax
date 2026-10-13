# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from .configuration import register_hf_config


register_hf_config()

from .bridge import DeepSeekV41Bridge  # noqa: E402
from .provider import DeepSeekV41ModelProvider  # noqa: E402


__all__ = ["DeepSeekV41Bridge", "DeepSeekV41ModelProvider"]
