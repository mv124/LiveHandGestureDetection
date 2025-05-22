# Live Hand Object Detection

A real-time hand detection and tracking application using OpenCV and MediaPipe.

## Features

- Real-time hand detection using webcam
- Dynamic circle tracking around detected hands
- Hand landmark visualization
- Smooth tracking with high confidence thresholds

## Requirements

- Python 3.11
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LiveObjectDetection.git
cd LiveObjectDetection
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

Run the hand detection script:
```bash
python src/hand_detection.py
```

- Press 'ESC' to exit the application
- Make sure your webcam is properly connected and accessible

## Project Structure

```
LiveObjectDetection/
├── src/                    # Source code
│   └── hand_detection.py   # Main application file
├── tests/                  # Test files
├── requirements.txt        # Project dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 