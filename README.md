# Emotion Detection Model
Loads the tensors provided by mediapipe to a pytorch model of the similar architecture and then finetunes the model on the `MER Dataset` for emotion detection using landmarks.

MER dataset is preprocessed the same way as in the Landmarks paper and the training requires the path to the reorganized dataset. Check the chin-chime repository for details.

Check the `exploratory.ipynb` model for the details.
The weights of the trained models are small enough that they are uploaded to github directly in the `model_weights` directory.

# mediapipe_pytorch
PyTorch implementation of Google's Mediapipe model. Iris Landmark model | Face Mesh Model. Check the original repository for the implementation details.


## Face Mesh Model
  facial_landmark folder contains the PyTorch implementation of paper Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs (https://arxiv.org/pdf/1907.06724.pdf)
  * For inference
      ```bash
      cd facial_landmark
      !python inference.py
      ```

## Iris Landmark Model
  iris folder contains the PyTorch implementation of paper Real-time Pupil Tracking from Monocular Video for Digital Puppetry (https://arxiv.org/pdf/2006.11341)
  * For inference
      ```bash
      cd iris
      !python inference.py
      ```

## Conversion Issues
    * TFLite uses slightly different padding compared to PyTorch.
    * Instead of using the padding parameter in the conv layer applying padding manually.
    * Change the padding value.
        * Misleading results
            * nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=True)
        * Correction
            * nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True)
                * x = nn.ReflectionPad2d((1, 0, 1, 0))(x) # Apply padding before convolution.
             
