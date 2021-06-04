# Final-Year-Project
University Work Submission Purposes
9.0 Code Guide

9.1 Generate Dataset
Code can be found in Generate_Images.ipynb and Organise_dataset.ipynb under FYP_code folder. Just enter folder paths of your object and background images in order to generate your own dataset. Both files are jupyter notebook files. Please read the methodology section on preparing dataset for more information.

9.2 Train Yolov5
In Yolo -> yolov5 folder
With your dataset created, run train.py using command prompt or Ananconda prompt which is preferred.
An example on how to run the train code.However it might be different based on your own problem. User might have more or less classes, weights trained for different objects and img sizes.
Train_Test_with_Raw.ipynb under yolo folder is a jupyter notebook file that provides illustration on how training works.
Please refer to ultralytics github link provided below for more detailed information.

9.3 Running full version application
Under yolo -> yolov5 folder, file my_RUN.py is the code file that is used for demos.
Image below provides an example on how it can be called. Once again it will be different for everyone depending on their problem to be solved.
Please refer to ultralytics github link provided below for more detailed information.
