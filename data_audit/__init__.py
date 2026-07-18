"""
DataAudit Framework

Author:
Md. Noman

Department:
Agriculture
Noakhali Science and Technology University (NSTU)

License:
AGPL-3.0 License

Copyright (c) 2026 Md. Noman
"""

from collections import namedtuple
from .accessor import DataAuditAccessor
from .ml import MLModule
from .report import summary_report, audit_report

_sid_tuple = namedtuple("sid", ["id", "value"])

def sid(id, value):
    """
    Creates a Specific ID (sid) object for manual data fixing.
    
    Args:
        id (int or str or tuple): The target to fix. Can be an Issue ID (int), 
            a column name (str), or a specific cell (row_index, col_name).
        value (any): The value to replace it with. Use "ignore" to skip fixing.
        
    Returns:
        namedtuple: A named tuple containing the id and value.
    """
    return _sid_tuple(id, value)

__all__ = ["DataAuditAccessor", "MLModule", "summary_report", "audit_report", "sid"]
print("INIT WAS EXECUTED!")
