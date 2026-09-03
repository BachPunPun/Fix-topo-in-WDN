# -*- coding: utf-8 -*-
"""
Cấu hình console Windows hỗ trợ UTF-8.

Cách dùng:
    import setup_console  # chỉ cần import, không cần gọi hàm
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")