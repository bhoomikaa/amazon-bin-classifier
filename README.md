Amazon Bin Object Classifier



This project builds a vision–language model that classifies and counts objects inside warehouse bins using CLIP (Contrastive Language–Image Pretraining) and EfficientNet. 

It demonstrates how modern AI models can perform zero-shot classification — recognizing concepts they haven’t been trained on — and how fine-tuning improves performance.



------------------------------------------------------------------



Project Overview



In large warehouses, manually identifying and counting items in storage bins is time-consuming. 

This project automates that process using two complementary approaches:



1\. Zero-Shot CLIP Classification – Uses OpenAI’s CLIP model to match image embeddings with text prompts like "a photo of a bin containing 3 items".

2\. Fine-Tuned EfficientNet – Trains a supervised classifier on labeled images to improve bin-counting accuracy.



The goal is to compare both methods and evaluate how well pretrained vision–language models generalize to real-world counting tasks.



------------------------------------------------------------------



Key Features



\- Image preprocessing (resize, normalize, augment)

\- CLIP-based image and text embedding generation

\- Zero-shot classification without custom training

\- EfficientNet fine-tuning for supervised learning

\- Evaluation metrics: accuracy, MAE, confusion matrix

\- Modular notebooks for preprocessing, embeddings, metrics, and training



------------------------------------------------------------------



Tech Stack



Language: Python 3.10+

Frameworks: PyTorch, OpenCLIP, timm

Libraries: NumPy, Pandas, scikit-learn, Matplotlib, tqdm

Hardware: Works with both CPU and GPU



------------------------------------------------------------------



Repository Structure



amazon-bin-classifier/

│

├── image\_preprocessing.ipynb              Preprocessing and augmentation

├── cLip\_img\_embeddings.ipynb              CLIP image embeddings

├── Text\_Embeddings.ipynb                  CLIP text embeddings

├── Object\_Counting\_Zero\_Shot\_CLIP.ipynb   Zero-shot evaluation

├── clip\_metrics\_compute.ipynb             Metric computation

├── CLIP\_Efficient\_Net\_Train.py            Fine-tuning EfficientNet

├── cleaned\_data.csv                       Sample labeled dataset

└── README.md                              Project documentation



------------------------------------------------------------------



How to Run



1\. Install dependencies:

&nbsp;  pip install -r requirements.txt



2\. Run preprocessing:

&nbsp;  jupyter notebook image\_preprocessing.ipynb



3\. Generate embeddings:

&nbsp;  jupyter notebook cLip\_img\_embeddings.ipynb



4\. Evaluate zero-shot classification:

&nbsp;  jupyter notebook Object\_Counting\_Zero\_Shot\_CLIP.ipynb



5\. (Optional) Fine-tune EfficientNet:

&nbsp;  python CLIP\_Efficient\_Net\_Train.py



------------------------------------------------------------------



Example Results



Model: CLIP (Zero-Shot)

Top-1 Accuracy: 78%

MAE: 1.3

Description: Uses natural-language prompts, no training



Model: EfficientNet-B0 (Fine-Tuned)

Top-1 Accuracy: 93%

MAE: 0.5

Description: Trained for 10 epochs on labeled data



------------------------------------------------------------------



Future Work



\- Add object detection to locate items within bins

\- Experiment with ViT or Swin Transformer architectures

\- Deploy inference as a REST API or Streamlit demo



------------------------------------------------------------------



Author



Bhoomikaa

A personal deep-learning project exploring vision–language models and warehouse automation.



------------------------------------------------------------------



License



MIT License — free for research and educational use.



