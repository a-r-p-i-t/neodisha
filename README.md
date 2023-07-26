## neodisha
# input parameters

1.weights_path : Model path of tray detection model

2.image : cv2.imread("path/of/image/file")

3.label_file : Path of file to save tray coordinates detected by tray detection model

4.model_path : Model path of object detection model

5.destination_path : Path of image file


# Output parametes

1.  Results Dictionary containing image name, tray coordinates, count_list of objects and empty places

2.  all tray patches are stored in working directory

3.  indivisual label files generated for each tray containing detection coordinates of objects and empty places in runs/detect/predict/labels





For each image,all trays can be visualised, count can be known by resultant dictionary

