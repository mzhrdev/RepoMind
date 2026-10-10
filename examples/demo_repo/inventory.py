"""A small in-memory inventory store for the RepoMind demo."""

from calculator import CalculatorError


class Inventory:
    """Tracks item quantities in a simple dictionary-backed store."""

    def __init__(self):
        self._items = {}

    def add_item(self, name, quantity):
        """Add quantity of an item, creating it if it does not exist."""
        if quantity < 0:
            raise ValueError("quantity cannot be negative")
        self._items[name] = self._items.get(name, 0) + quantity

    def remove_item(self, name, quantity):
        """Remove quantity of an item, raising if there is not enough stock."""
        current = self._items.get(name, 0)
        if quantity > current:
            raise CalculatorError("not enough stock to remove")
        self._items[name] = current - quantity

    def total_items(self):
        """Return the total quantity across all tracked items."""
        return sum(self._items.values())
