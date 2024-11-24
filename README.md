
# Video Stabilization using Lucas-Kanade Method

This repository contains a detailed implementation of video stabilization using the Lucas-Kanade optical flow method with OpenCV in Python. The goal is to reduce unwanted shakiness in video footage and provide a smoother, more stable output.

The Lucas-Kanade method is used for tracking keypoints between consecutive frames, allowing us to estimate the camera movement and subsequently stabilize the video. This implementation also involves trajectory smoothing to ensure a steady output.

## Requirements

To run the code, you need:

- Python 3.x

It's recommended to use a virtual environment to manage dependencies. You can create and activate a virtual environment using the following commands:

```sh
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

After activating the virtual environment, install the required packages:

```sh
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```sh
git clone https://github.com/AdiGhadge/Video-Stabilization.git
```

2. Navigate to the project directory:

```sh
cd Video-Stabilization
```

3. Activate the virtual environment (if not already activated):

```sh
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

4. Run the script with the input video file:

```sh
python video_stabilization.py
```

The script will read the input video ('motorcycles.mp4' by default) and output a stabilized version ('motorcycles_stabilized.mp4').

Alternatively, you can modify the script to use different input and output file paths directly in the code.

## Example

```sh
python video_stabilization.py
```

## Contributors

- [Soham Sant](https://github.com/SohamSant)
- [Aditya Ghadge](https://github.com/AdiGhadge)
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
