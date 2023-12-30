import cv2
class config: # configurations for the camera, display, and user
    
    # Default configurations.
    # Currently configured for display and webcam of Dell Precision 5570 Mobile Workstation

    # Camera ID and resolution
    cameraname = "/dev/video0"
    camw = 1280 
    camh = 720

    # These values can be found on the manufacturer's datasheet of your camera.
    # This assumes that your camera has no distortion of the image plane
    # because these types of distortion are not common in laptop webcams
    
    HFOV = 73.739 # horizontal FOV in degrees.
    VFOV = 38.5 # vertical field of view

    # Display resolution.
    dispw = 3840 
    disph = 2400 

    # Measurements of real world space. All are in inches.

    screen_diagonal = 15.6 #in inches
    camera_y_offset = 4.25 # vertical distance between the center of the screen and the lens of the webcam.
    camera_x_offset = 0 # horizontal distance parallel to screen
    camera_z_offset = 0 # horizontal distance normal to screen

    # Measurements of your face. See reference image for details.
    eedist = 4.3
    endist = 3

    def configcamera(self, cap):
        # Here is where you configure the video capture settings.
        # Edit configurations like exposure based on your setup
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camw)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camh)
        ret_val, frame = cap.read()


    