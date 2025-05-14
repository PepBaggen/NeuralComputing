<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#model-overview">Model overview</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#support">Support</a></li>
    <li><a href="#authors-and-acknowledgment">Acknowledgments</a></li>
    <li><a href="#project-status">Project status</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This project implements a food image classifier using a Convolutional Neural Network (CNN) using PyTorch. 

### Model overview
The model is trained from scratch on a folder-based dataset consisting of 91 different food categories. Before training, the data is augmented, i.e. transformed in several random ways, and normalized using per channel mean and SD pixel values.

The CNN consists of three convolutional blocks of depth 128, 256 and 512 respectively. Every block contains two 2D convolution layers, each followed by batch normalization and a ReLU activation function, and ends with 2D max pooling. Finally, the model ends by applying a linear transformation and dropout before connecting to the output layer.

The final model reached 55% accuracy after 100 epochs of training. We apply a scheduler and a plateau that decrease the learning rate if the difference in accuracy is too small. The hyper-parameters consist of a batch size of 32, starting learning rate of 0.0005, weight decay of 0.0005, plateau accuracy threshold of 0.1 and epoch patience of 2. After every epoch, we compute the cross-entropy loss and model accuracy.

We conclude the project by simulating a user uploading 10 test set pictures. We return the predicted model categories and an LLM description of the user's food preference using the OpenAI API.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

Download the assignment_NC2425.ipynb file and navigate in Terminal to the folder containing the downloaded file. For example:

```sh
cd C:\Users\user\Downloads\project
```

If you do not have access to the original train and test folders of food images, make sure your own dataset has the following structure to readily work with the code:
```sh
.
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── ...
│   └── class_n/
├── test/
│   ├── class_1/
│   └── ...
```

The following Python libraries are required to run the notebook.
* Pandas
* PyTorch 2.6
* torchvision
* numpy
* pillow
* os
* scikit-learn
* matplotlib

Download them from the requirements.txt file using:
```sh
pip install -r requirements.txt
```

Now the notebook should be ready for execution. Either run every cell at a time or the whole notebook at once.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage
This notebook can be run fully on the same dataset to replicate our reported model accuracy. It can also be run on new datasets for comparisons across different image classification needs. However, note that the current CNN architecture was fine-tuned on the given food image dataset, meaning the model performance is likely poor on new types of images.

## Support
For questions please email s2995387@vuw.leidenuniv.nl.

## Authors and acknowledgment
Pepijn Baggen s3193993

Foppe van Berkel s2995387

Max Gloesener s3955435

Mylène Butselaar s2723069

## Project status
Finished