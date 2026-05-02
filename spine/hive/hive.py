"""Hive - Durable task tracking (swarm-tools pattern)."""

import json
import os
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class Cell:
    """A durable task record (hive cell)."""
    cell_id: str
    title: str
    type: str = "task"
    status: str = "pending"
    assignee: str = ""
    phase: str = ""
    priority: str = "medium"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    file_reservation: Optional[dict[str, Any]] = None
    swarm_events: list[str] = field(default_factory=list)
    result: Optional[dict[str, Any]] = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Hive:
    """
    Durable task tracking using the swarm-tools pattern.
    
    Cells are git-syncable task records that survive across sessions.
    """
    
    def __init__(self, path: str = ".spine/state/hive"):
        self.path = path
        self._cells: dict[str, Cell] = {}
        os.makedirs(path, exist_ok=True)
        self._load()
    
    def _load(self):
        """Load cells from disk."""
        cells_path = os.path.join(self.path, "cells.json")
        if os.path.exists(cells_path):
            with open(cells_path) as f:
                data = json.load(f)
                for cell_data in data.get("cells", []):
                    cell = Cell(**cell_data)
                    self._cells[cell.cell_id] = cell
    
    def _save(self):
        """Save cells to disk."""
        cells_path = os.path.join(self.path, "cells.json")
        data = {
            "version": "1.0",
            "cells": [c.to_dict() for c in self._cells.values()]
        }
        with open(cells_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def create_cell(self, **kwargs) -> Cell:
        """Create a new hive cell."""
        cell_id = kwargs.get("cell_id") or f"cell_{len(self._cells) + 1:03d}"
        cell = Cell(cell_id=cell_id, **kwargs)
        self._cells[cell_id] = cell
        self._save()
        return cell
    
    def get_cell(self, cell_id: str) -> Optional[Cell]:
        """Get a cell by ID."""
        return self._cells.get(cell_id)
    
    def update_cell(self, cell_id: str, **updates) -> Optional[Cell]:
        """Update a cell's properties."""
        if cell_id in self._cells:
            cell = self._cells[cell_id]
            for key, value in updates.items():
                if hasattr(cell, key):
                    setattr(cell, key, value)
            self._save()
            return cell
        return None
    
    def list_cells(self, status: Optional[str] = None) -> list[Cell]:
        """List all cells, optionally filtered by status."""
        cells = list(self._cells.values())
        if status:
            cells = [c for c in cells if c.status == status]
        return cells