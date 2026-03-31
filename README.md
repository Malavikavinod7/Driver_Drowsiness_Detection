# Real-Time Drowsy Driver Detection System
OverviewFatigue-related highway accidents account for roughly one-third of all highway fatalities in India (NHAI). This project provides a transparent, hardware-agnostic solution to detect driver impairment in real-time.Unlike "black-box" neural networks, this system uses Geometric Feature Engineering to monitor facial landmarks, specifically focusing on Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to trigger life-saving alerts.

Key Features:
Zero Extra Hardware: Runs on a standard laptop webcam without requiring a GPU.
EAR Monitoring: Detects eye closure held for $\approx$ 2 seconds (microsleep detection).
MAR Monitoring: Identifies yawning thresholds to catch early-stage fatigue.
Real-Time Audio Alerts: Immediate feedback when thresholds are breached.
MediaPipe Integration: Uses a 468-point face mesh for high-fidelity landmark tracking.

Performance:
Balanced Accuracy - 73%
Methodology       - Pixel-based feature approach
Optimization      - Built to handle image quality constraints and varied lighting


How It Works:
The system calculates ratios based on Euclidean distances between specific eyelid and lip landmarks.
 Eye Aspect Ratio (EAR)The EAR formula determines if the eye is open or closed Mouth Aspect Ratio (MAR)The MAR monitors the vertical opening of the mouth. If the ratio exceeds a pre-defined threshold, a yawn is registered.


Prerequisites:
Python 3.8+A functional webcamInstallationClone the repository:Bashgit clone https://github.com/yourusername/drowsy-driver-detection.git
cd drowsy-driver-detection
Install dependencies:Bashpip install opencv-python mediapipe numpy pygame scipy
Run the application:Bashpython main.py


