"""Shared constants for the rotation detection pipeline."""

CARDINAL_ANGLES: tuple[int, int, int, int] = (0, 90, 180, 270)

ANGLE_TO_INDEX: dict[int, int] = {angle: idx for idx, angle in enumerate(CARDINAL_ANGLES)}
INDEX_TO_ANGLE: dict[int, int] = {idx: angle for angle, idx in ANGLE_TO_INDEX.items()}

WHITE_RGB: tuple[int, int, int] = (255, 255, 255)
