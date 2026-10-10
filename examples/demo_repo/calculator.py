"""A tiny synthetic calculator module used to demo RepoMind."""


class CalculatorError(Exception):
    """Raised when an invalid calculator operation is requested."""


def add(a, b):
    """Return the sum of two numbers."""
    return a + b


def subtract(a, b):
    """Return the difference of two numbers."""
    return a - b


def divide(a, b):
    """Divide a by b, raising CalculatorError on division by zero."""
    if b == 0:
        raise CalculatorError("Cannot divide by zero")
    return a / b


def apply_operation(op, a, b):
    """Dispatch an operation name to the matching calculator function."""
    operations = {
        "add": add,
        "subtract": subtract,
        "divide": divide,
    }
    if op not in operations:
        raise CalculatorError(f"Unknown operation: {op}")
    return operations[op](a, b)
