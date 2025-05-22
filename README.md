# Live Hand Gesture Detection

A real-time hand gesture detection and recognition application using OpenCV and MediaPipe.

## Features

- Real-time hand detection and tracking using webcam
- Hand landmark detection and visualization
- Gesture recognition capabilities
- Smooth tracking with high confidence thresholds
- Real-time visualization of hand movements

## Requirements

- Python 3.11
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LiveHandGestureDetection.git
cd LiveHandGestureDetection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the hand gesture detection script:
```bash
python src/hand_detection.py
```

- Press 'ESC' to exit the application
- Make sure your webcam is properly connected and accessible
- Position your hand clearly in front of the camera for optimal detection

## Project Structure

```
LiveHandGestureDetection/
├── src/                    # Source code
│   └── hand_detection.py   # Main application file
├── tests/                  # Test files
├── requirements.txt        # Project dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Features in Detail

- **Hand Detection**: Real-time detection of hands using MediaPipe's hand tracking solution
- **Landmark Detection**: Detection of 21 hand landmarks for precise hand pose estimation
- **Gesture Recognition**: Recognition of various hand gestures and poses
- **Real-time Visualization**: Live display of hand tracking and gesture recognition results

## License

This project is licensed under the MIT License - see the LICENSE file for details. 