# Feature-Matching

This is inspired by inbuilt functioanality in OpenCV on Flann Based Matcher. The code was written for a competition where a yellow and black caution tape needed to be detected and it's distance from the camera had to be determined. The camera was mounted to a drone and  distance is calculated knowing the height of the drone and the camera angles. This is similar to template matching but since features are rotation invariant, the tape could be detected at any angle in real-time.

Model.jpg is the image which contains the template of what is to be detected. This code is written in Linux and to run the code, download the zip and extract it. Go to the folder in the terminal and do:

- cmake .
- make

and if everthing runs without errors then run the executable by typing:

- ./features testImage.jpeg

The output window should look something like this:
