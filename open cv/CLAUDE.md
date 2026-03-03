# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Applications

```bash
# Run advanced OpenCV exercises (face detection, edge detection, color detection)
python app.py

# Run basic OpenCV operations (video frame extraction, image processing)
python "open_cv operational.py"
```

## Project Structure

This OpenCV computer vision project contains two main Python files:
- `app.py` - Advanced computer vision applications with practical exercises
- `open_cv operational.py` - Basic OpenCV operations and video processing

The project uses sample images (`Mohith.jpeg`, `image_boys.png`, `lots_of_people.jpg`) and a video file (`video.mov`) for demonstrations.

## Dependencies

- opencv-python==4.9.0.80
- numpy==1.26.4

The virtual environment (`venv/`) contains all necessary packages for OpenCV operations.