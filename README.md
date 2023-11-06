# Final Year Project Submission 2022

## Instructions to run

*Tested On Linux Only*

- Ensure Nvidia drivers, docker, nvidia-container-toolkit/nvidia-docker2 is installed

### Steps to Build Docker Image

`git clone https://github.com/jeffreypaul15/FinalYearProject`

`cd FinalYearProject`

`docker build -t fyp:latest .`

### Steps to run the Docker Image

`docker run -it --name fyp_host --gpus=all -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,display -v /tmp/.X11-unix:/tmp/.X11-unix --env NVIDIA_DISABLE_REQUIRE=1 --device=/dev/video2 --privileged --network host -v /some/path/on/host/:/workspace`

### Test changes by running one of the videos

On Linux systems using xorg, You have to execute this once after boot to allow permissions for passing display
`xhost +`

In the container, start the webui using python3 http server

`cd /workspace/FinalYearProject/front_end`
`python3 -m http.server`

A web server is started and you can navigate to localhost:8000 in your browser, sign up and login

To run accident detection on a video

`cd /workspace/FinalYearProject/back_end/server_code/`
`python3 server.py --path /path/to/a/video`
