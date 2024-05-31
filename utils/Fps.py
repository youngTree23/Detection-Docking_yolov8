import time
import cv2


def calculate_fps(previous_time, visualization, frame=None):
    current_time = time.time()
    elapsed_time = current_time - previous_time
    fps = 1 / elapsed_time if elapsed_time else float('inf')
    previous_time = current_time

    if visualization and frame is not None:
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return fps, previous_time
