#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 WorkflowManager 是否有 get_differentiation_history 方法
"""

import sys
sys.path.insert(0, '/home/wx/AFlow')

from scripts.utils.workflow_manager import WorkflowManager
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.graph_utils import GraphUtils

# 初始化依赖
root_path = "workspace/MIXED"
data_utils = DataUtils(root_path, None, None)
graph_utils = GraphUtils(root_path, None, None, None)

# 创建 WorkflowManager 实例
workflow_manager = WorkflowManager(
    root_path=root_path,
    data_utils=data_utils,
    graph_utils=graph_utils
)

# 检查方法是否存在
print(f"WorkflowManager has get_differentiation_history: {hasattr(workflow_manager, 'get_differentiation_history')}")

# 尝试调用
if hasattr(workflow_manager, 'get_differentiation_history'):
    try:
        history = workflow_manager.get_differentiation_history()
        print(f"✓ Method called successfully")
        print(f"  History: {history}")
    except Exception as e:
        print(f"✗ Method call failed: {e}")
else:
    print(f"✗ Method not found")
    print(f"Available methods: {[m for m in dir(workflow_manager) if not m.startswith('_')]}")
