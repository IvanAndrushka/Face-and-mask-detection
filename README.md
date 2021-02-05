# Face-and-mask-detection

This is a tutorial project using opencv and neural networks to perform face detection and check whether a person is wearing a mask or not.

The main files are:  
-train_mask_detector.py: train a MobileNetV2 network to classify whether a person is using a mask or not. The training set is in the folder dataset.  
-detect_mask_image.py: detects faces using a mtcnn network and classifies the mask usage on with the MobileNetV2 model.  
-detect_mask_video.py: does the same as the above but with a video. (the algorithm is pretty slow though, so you'll only get a few fps at most)

The other files are just small snippets of code I used to learn :).

Here's an example of the program in action (run with python .\detect_mask_image.py --image beautiful.jpg)

![alt text](https://github.com/IvanAndrushka/Face-and-mask-detection/blob/Master/mask_detected.png?raw=true)
