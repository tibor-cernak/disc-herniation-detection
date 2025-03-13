from typing import Tuple


class Annotation:
    def __init__(
        self, slice: int, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> None:
        """
        Represents an annotation for a specific slice in an MRI volume.

        This annotation is a line defined by two points (p1 and p2) in normalized coordinates.

        Args:
            slice (int): The slice index where the annotation is located.
            p1 (Tuple[float, float]): The starting point of the line.
            p2 (Tuple[float, float]): The ending point of the line.
        """
        self.slice: int = int(slice)
        self.p1: Tuple[float, float] = p1
        self.p2: Tuple[float, float] = p2

    def __str__(self) -> str:
        return f"Annotation at slice {self.slice} | Line: {self.p1} -> {self.p2}"
