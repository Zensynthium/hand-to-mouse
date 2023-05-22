Use your hand as a mouse! Utilizes a webcam/camera, OpenCV, and Mediapipe Hands, all with Python!

While developing this and setting it up for use, I used a python virtual environment as instructed in the Mediapipe getting started docs for Python (linked below the code):

python3 -m venv mp_env && source mp_env/bin/activate
pip install mediapipe pynput mss

NOTE: The first command can be different depending on how Python is configured on your machine. For example, with only Python 3 on your machine and on windows, the command will be as follows:

python -m venv mp_env && source mp_env/scripts/activate

Different operating systems may create a slightly different file path and you'll have to check where the activate script is.

After succesfully setting up the virtual environment and installing the mediapipe package, mediapipe related code should work.

Mediapipe Hands Docs:
https://google.github.io/mediapipe/solutions/hands.html

Mediapipe Getting Started (Python):
https://google.github.io/mediapipe/getting_started/python.html
