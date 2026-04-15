"""
data/schema.py

Defines the Task dataclass used throughout the system.
Keeping this separate means every module imports from one place,
and changing a field name only requires an edit here.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Task:
    """
    Represents a single study task for a student.

    Fields
    ------
    task_id : str
        Unique identifier, e.g. "TASK_001".
    name : str
        Human-readable name, e.g. "Data Structures Assignment 3".
    deadline : datetime
        When the task must be submitted.
    estimated_hours : float
        How many hours the student thinks this task will take.
    difficulty : int
        Self-rated difficulty on a 1-5 scale (1=trivial, 5=extremely hard).
    priority : int
        Student-assigned importance on a 1-5 scale (1=low, 5=critical).
    available_hours : float
        Hours the student has free before the deadline.
    subject : str
        Subject area, e.g. "Mathematics". Useful for filtering later.
    completed : bool
        Whether the task has been finished. Completed tasks are excluded
        from ranking automatically.
    """

    task_id: str
    name: str
    deadline: datetime
    estimated_hours: float
    difficulty: int          # 1 to 5
    priority: int            # 1 to 5
    available_hours: float
    subject: str = "General"
    completed: bool = False

    def hours_until_deadline(self, reference_time: datetime = None) -> float:
        """
        Returns how many hours remain until the deadline.
        Negative values mean the deadline has already passed.

        Parameters
        ----------
        reference_time : datetime, optional
            The point in time to measure from. Defaults to now.
        """
        if reference_time is None:
            reference_time = datetime.now()
        delta = self.deadline - reference_time
        return delta.total_seconds() / 3600.0

    def is_overdue(self, reference_time: datetime = None) -> bool:
        return self.hours_until_deadline(reference_time) < 0

    def __repr__(self):
        return (
            f"Task(id={self.task_id!r}, name={self.name!r}, "
            f"difficulty={self.difficulty}, priority={self.priority})"
        )
