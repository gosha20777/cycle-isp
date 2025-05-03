# Learned Lightweight Smartphone ISP with Unpaired Data @ CVPRW 2025
### The IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2025, Nashville

[Andrei Arhire](https://scholar.google.com/citations?user=BYkEZGFPq1wC&hl=ro), [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)

----

The **Image Signal Processor (ISP)** is a fundamental component in modern smartphone cameras responsible for conversion of RAW sensor image data to RGB images with a strong focus on perceptual quality. A difficult and costly step when developing a learned ISP is the acquisition of pixel-wise aligned paired data that maps the raw captured by a smartphone camera sensor to high-quality reference images.

We address this challenge by proposing **a novel training method** for a learnable ISP that eliminates the need for direct correspondences between raw images and ground-truth data with matching content. Our **unpaired approach** employs a multi-term loss function guided by adversarial training with multiple discriminators processing feature maps from pre-trained networks to maintain content structure while learning color and texture characteristics from the target RGB dataset.

<img src="media/narchitecture.png" alt="architecture" width="800"> 

Compared to paired approaches, our strategy does not require the costly and time-consuming acquisition of paired datasets and is not affected by pixel misalignment caused by factors such as viewpoint differences, dynamic elements, or artifacts introduced by alignment algorithms.

<img src="media/nsamples.png" alt="samples" width="800"> 


----

### Prerequisites

- Python and all packages listed in [`requirements.txt`](./requirements.txt)
- Pytorch and CUDA CuDNN
- Nvidia GPU
- Download dataset [Fujifilm UltraISP](https://github.com/gosha20777/mai-25/releases/tag/1.0.0) or [Zurich RAW to RGB](http://data.vision.ee.ethz.ch/ihnatova/public/zr2d/Zurich-RAW-to-DSLR-Dataset.zip)
- Pre-trained [VGG-19 model](https://polybox.ethz.ch/index.php/s/7z5bHNg5r5a0g7k)


----

### How to Run

- Training
  
    Most of the hyperparameters can be set in ```config.py```.
  
    To start training and save model checkpoints after a specified number of updates, run the ```fit.py``` script:
  
```
    python fit.py
```
- Validation
  
  Models saved during training can be evaluated in parallel on the validation set. The model achieving the highest fidelity score can then be selected for evaluation on the test data.   
  
  ```process_set.py``` accepts two parameters, ```n``` and ```step```, and evaluates only those checkpoint models whose index, when divided by ```n```, yields a remainder that is a multiple of ```step```. For each such remainder, ```process_set.py``` processes all corresponding models and calls ```inference.py``` to perform inference on the validation data and compute performance metrics.

  For example, the command below will create 4 tmux sessions that simultaneously evaluate models whose index modulo 100 results in 0, 25, 50, or 75.

  ```
  python process_set.py --n 100 --step 25
  ```
  
- Testing

  Inference can be performed using ```test_model.py``` script.

  ```
  python test_model.py
  ```
  
----

### Pre-trained models

We provide pre-trained models and inference scripts in [`pretrained/`](./pretrained/) directory, corresponding to the models listed in Tables 1 and 2 of the paper.

### Citation

If you find this work useful for your research, please cite our paper:

```
@InProceedings{arhire2025learned,
    author    = {Andrei Arhire and Radu Timofte},
    title     = {Learned Lightweight Smartphone ISP with Unpaired Data},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2025}
}
```

### Acknowledgments

We are grateful for the helpful resources provided by the following projects:

- [pyiqa toolbox](https://github.com/chaofengc/IQA-PyTorch)
- [DPED](https://github.com/aiff22/DPED)
- [R3GAN](https://github.com/brownvc/R3GAN)
- [LAN](https://github.com/draimundo/LAN)

**Contact** andreiarhire2708@gmail.com
