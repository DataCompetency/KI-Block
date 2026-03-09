# Neural Networks and Deep Learning
## Greifswald, March 9-10, 2026
Go to [Moodle](https://moodle.uni-greifswald.de/course/view.php?id=5405) for
 - Q & A forum
 - feedback
 - announcements
 - quiz

## Get Started

1. Login to [Moodle](https://moodle.uni-greifswald.de/course/view.php?id=5405) (requires registration). Leave the Q & A tab open in your browser.
2. Login at [apphub.wolke.uni-greifswald.de](https://apphub.wolke.uni-greifswald.de/), choose **Computing -> Deep Learning -> Select -> Frontend Code -> Select**, set the time to approximately **8 hours**, then click **Start**.
3. Once the session is booted, click the **Open** button.
4. In VSCode, open a terminal (`` Ctrl+` ``) and run: `git clone https://github.com/DataCompetency/KI-Block.git`
5. Install required packages in the same terminal:

   ```bash
   /opt/conda/envs/d2l/bin/pip install pandas torchsummary "opencv-python-headless==4.8.0.76" "numpy==1.23.5"
   ```

6. `cd` into the cloned folder, then open it in VSCode via **File -> Open Folder**.
7. Navigate to the exercises folder and open **es0.ipynb**.
8. Select **d2l (Python 3.11.11)** as the Python kernel.

> **Note:** Instructions last updated March 9, 2026 — subject to change.
 
## Course Overview
In this course you will get practical knowledge to perform general machine learning and, in particular, computer vision tasks with TensorFlow. Moreover, you will get the necessary theoretical background to troubleshoot when transferring the knowledge to solve own problems. The class will be hands-on and interactive. A device with a browser, keyboard and a VPN connection to the Rechenzentrum is required.

### (Stochastic) Gradient Descent

<img src="figs/intro-sgd.png" width="800"/>

### Linear Regression with TensorFlow 2

<p>
  <img src="figs/intro-lr.png" width="400"/>
  <img src="figs/intro-qr.png" width="400"/>
</p>


### Expore the Effect of Critical Parameters

<img src="figs/intro-sliders.png" width="700"/>


### Derive and Optimize

<p align="center">
 <img src="exercises/dy_dx.png" width="200"/>
</p>

### Fully-Connected Neural Network for Regression

<p align="center">
<img src="exercises/nn8-8.png" width="600"/>
<img src="figs/intro-nn-reg.png" width="800"/>
</p>


### Convolutional Neural Networks (Architectures, Filters, Feature Maps)

<img src="exercises/cnn-ngon.png" alt="cnn architecture" width="800"/>
<img src="figs/intro-filters.png" width="600"/>
<img src="exercises/exciting-patches.png" width="500"/>
<img src="figs/intro-fool-ex.png" width="600"/>


### Sequence and Foundation Models for DNA

<img src="figs/dna-onehot.png" width="600"/>


