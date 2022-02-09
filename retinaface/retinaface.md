# Retina Face 

[RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/retinaface) 
is a practical single-stage SOTA face detector which is initially introduced in [arXiv](https://arxiv.org/abs/1905.00641) (2019) 
and [IEEE](https://ieeexplore.ieee.org/document/9157330) (2020). 

### Data 
__WIDER Face Dataset__ consists of *32303* images, *393703* face bounding boxes with a high degree of variability in scale, pose, expression, occlusion and illumination.

Data is additionally annotated 5 facial landmarks (eye centers, nose tip and mouth corners).

Data is divided into 5 levels (according to how difficult it is to annotate landmarks on the face)

<details>
  <summary> 5 Levels Face Data Description </summary>
  <center>
  
  |  Level  |  Face Number |  Criterion | 
  |---|---|---|
  |  1 (easiest)  |  4127  |  Easy to determine 68 landmarks  |
  |  2   |  12636  |  Can determine 68 landmarks  |
  |  3  |  38140  |  Easy to determine 5 landmarks  |
  |  4  |  50024  |  Can determine 5 landmarks  |
  |  5 (hardest)  |  94095  |  Determine by contexts  |
  
  </center>
</details>

In total, we have annotated 84.6k faces on the training set and 18.5k faces onthe validation set <br> 
(According to the paper, ratio of `train:val:test` is `40:10:50`, the exact ratio is a question)

<img src="./img/5_levels_annotatable.png" alt="5_levels_annotatable" width="500"/>
