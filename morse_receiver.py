import cv2
import numpy as np
import time
from collections import deque

"""
Morse Receiver via Webcam

This program reads LED blinks from a webcam and decodes them into Morse code.
It supports automatic camera detection, brightness threshold control, ROI control,
Morse decoding logic, message reset, and real-time UI updates.

Press:
- 'R' to reset the decoded message
- 'ESC' to exit safely
"""

# -------------------------------------------------------------------
# Morse Timing (must match the Arduino transmitter exactly)
# -------------------------------------------------------------------
DOT_MS = 200
DASH_MS = 600
GAP_SYMBOL_MS = 200
GAP_LETTER_MS = 600
GAP_WORD_MS = 1400

# Decoding thresholds
DOT_DASH_THRESHOLD_MS = (DOT_MS + DASH_MS) // 2
LETTER_GAP_MS = GAP_LETTER_MS
WORD_GAP_MS = GAP_WORD_MS

# Brightness detection
DEFAULT_BRIGHTNESS_THRESHOLD = 100
ROI_SIZE = 120
BRIGHTNESS_SMOOTHING_WINDOW = 5

# -------------------------------------------------------------------
# Morse Dictionary
# -------------------------------------------------------------------
MORSE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9'
}

# -------------------------------------------------------------------
# Camera Auto-detection
# -------------------------------------------------------------------
BACKENDS = [
    cv2.CAP_MSMF,   # Best for Windows
    cv2.CAP_DSHOW,
    cv2.CAP_ANY,
    cv2.CAP_VFW,
]

def auto_detect_camera():
    """Try multiple OpenCV backends and indices to detect a working camera."""
    print("Detecting available webcam...")
    for backend in BACKENDS:
        for index in range(4):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                print(f"Camera found at index {index} using backend {backend}")
                cap.release()
                return index, backend
    raise RuntimeError("No available camera detected.")


# -------------------------------------------------------------------
# Morse Decoder Class
# -------------------------------------------------------------------
class MorseDecoder:
    """Handles Morse symbol decoding based on ON/OFF timings."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all decoding buffers and reset state."""
        self.led_on = False
        self.last_change = time.time()
        self.current_symbol = ""
        self.current_word = ""
        self.full_message = ""
        self.last_letter = ""
        self.last_word = ""
        print("✨ Decoder reset!")

    def update(self, is_on):
        """
        Process LED ON/OFF transitions.
        Determines whether each ON period is dot or dash 
        and handles letter/word gaps.
        """
        now = time.time()
        duration_ms = (now - self.last_change) * 1000.0

        # No change → nothing to do
        if is_on == self.led_on:
            return

        # Update time reference
        self.last_change = now

        # LED switched OFF → end of dot/dash
        if self.led_on:
            if duration_ms < DOT_DASH_THRESHOLD_MS:
                self.current_symbol += "."
            else:
                self.current_symbol += "-"

        # LED switched ON → detect gap type
        else:
            if duration_ms >= WORD_GAP_MS:
                self.finalize_letter()
                self.finalize_word()
            elif duration_ms >= LETTER_GAP_MS:
                self.finalize_letter()

        self.led_on = is_on

    def finalize_letter(self):
        """Convert accumulated dots/dashes into a letter."""
        if not self.current_symbol:
            return
        letter = MORSE_DICT.get(self.current_symbol, '?')
        self.current_word += letter
        self.last_letter = letter
        self.current_symbol = ""

    def finalize_word(self):
        """Append decoded word to the message."""
        if not self.current_word:
            return
        if self.full_message:
            self.full_message += " "
        self.full_message += self.current_word
        self.last_word = self.current_word
        self.current_word = ""


# -------------------------------------------------------------------
# UI Helpers
# -------------------------------------------------------------------
def draw_text(frame, lines, x=10, y=20, dy=22):
    """Draw multiple UI text lines on screen."""
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x, y + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# -------------------------------------------------------------------
# Main Program
# -------------------------------------------------------------------
def main():
    try:
        cam_index, backend = auto_detect_camera()
    except Exception as e:
        print("ERROR:", e)
        return

    try:
        cap = cv2.VideoCapture(cam_index, backend)
        if not cap.isOpened():
            raise RuntimeError("Camera failed to open.")
    except Exception as e:
        print("ERROR:", e)
        return

    window = "Morse Receiver - Press ESC to Exit, R to Reset"
    try:
        cv2.namedWindow(window)
    except Exception:
        print("ERROR: Cannot create OpenCV window.")
        return

    # Trackbars
    cv2.createTrackbar("Threshold", window, DEFAULT_BRIGHTNESS_THRESHOLD, 255, lambda x: None)
    cv2.createTrackbar("ROI Size", window, ROI_SIZE, 300, lambda x: None)

    decoder = MorseDecoder()
    brightness_buffer = deque(maxlen=BRIGHTNESS_SMOOTHING_WINDOW)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Camera disconnected.")
                break

            h, w = frame.shape[:2]
            thresh = cv2.getTrackbarPos("Threshold", window)
            roi_size = max(40, cv2.getTrackbarPos("ROI Size", window))

            # ROI around center
            cx, cy = w // 2, h // 2
            s = roi_size // 2
            x1, y1 = max(0, cx - s), max(0, cy - s)
            x2, y2 = min(w, cx + s), min(h, cy + s)

            roi = frame[y1:y2, x1:x2]

            # Brightness
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            brightness_buffer.append(avg_brightness)
            smooth = np.mean(brightness_buffer)

            # LED ON/OFF
            led_on = smooth > thresh
            decoder.update(led_on)

            # Draw the ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0) if led_on else (0, 0, 255), 2)

            # Reset info
            cv2.putText(frame, "Press 'R' to Reset", (w - 260, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            # UI text
            draw_text(frame, [
                f"Brightness: {smooth:.1f}",
                f"Threshold: {thresh}",
                f"LED: {'ON' if led_on else 'OFF'}",
                f"Symbol: {decoder.current_symbol}",
                f"Letter: {decoder.last_letter}",
                f"Word: {decoder.current_word}",
                f"Message: {decoder.full_message}"
            ])

            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:     # ESC
                break
            elif key in (ord('r'), ord('R')):
                decoder.reset()

            # Auto finalize idle words
            now = time.time()
            if not decoder.led_on and (now - decoder.last_change) * 1000 > WORD_GAP_MS:
                decoder.finalize_letter()
                decoder.finalize_word()

    except Exception as e:
        print("Unexpected ERROR:", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nFINAL MESSAGE:", decoder.full_message)


if __name__ == "__main__":
    main()
