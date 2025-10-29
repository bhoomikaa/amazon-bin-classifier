\# 🧠 Amazon Bin Object Classifier  



This project builds a \*\*vision–language model\*\* that classifies and counts objects in warehouse bins using \*\*CLIP (Contrastive Language–Image Pretraining)\*\* and \*\*EfficientNet\*\*.  

It demonstrates how modern multimodal AI models can perform \*\*zero-shot classification\*\* — recognizing concepts they’ve never been explicitly trained on — and how fine-tuning can further improve performance.



---



\## 🔍 Project Overview  



In large warehouse environments, sorting and identifying the number or type of items inside storage bins is a repetitive task that can benefit from automation.  

This project explores two complementary approaches:  



1\. \*\*Zero-Shot CLIP Classification\*\* – Using OpenAI’s CLIP model to match image embeddings with text prompts like “a photo of a bin containing 3 items.”  

2\. \*\*Fine-Tuned EfficientNet\*\* – Training a supervised classifier on labeled bin images to improve counting accuracy.  



The goal is to compare both methods and understand how well pretrained vision-language models generalize to object-counting problems.



---



\## ⚙️ Key Features  

\- 🧩 Image preprocessing pipeline (resize, normalize, augment).  

\- 🔡 CLIP-based embedding generation for images and text.  

\- 🤖 Zero-shot classification without any training.  

\- 🧠 EfficientNet fine-tuning for supervised learning.  

\- 📊 Evaluation metrics: accuracy, mean absolute error (MAE), and confusion matrices.  

\- 💾 Modular notebooks for each stage — preprocessing, embeddings, metrics, and model training.



---



\## 🧰 Tech Stack  

\- \*\*Language:\*\* Python 3.10+  

\- \*\*Libraries:\*\* PyTorch, OpenCLIP, timm, NumPy, Pandas, scikit-learn, Matplotlib, tqdm  

\- \*\*Hardware:\*\* CPU / GPU compatible  



---



\## 📁 Repository Structure  

```

amazon-bin-classifier/

├── image\_preprocessing.ipynb        # Preprocessing and augmentation

├── cLip\_img\_embeddings.ipynb        # CLIP image embeddings

├── Text\_Embeddings.ipynb            # CLIP text embeddings

├── Object\_Counting\_Zero\_Shot\_CLIP.ipynb # Zero-shot experiments

├── clip\_metrics\_compute.ipynb       # Metrics \& evaluation

├── CLIP\_Efficient\_Net\_Train.py      # Fine-tune EfficientNet

├── cleaned\_data.csv                 # Sample labeled data

└── README.md                        # Project documentation

```



---



\## 🚀 How to Run  



1\. \*\*Install dependencies\*\*

&nbsp;  ```bash

&nbsp;  pip install -r requirements.txt

&nbsp;  ```



2\. \*\*Run preprocessing\*\*

&nbsp;  ```bash

&nbsp;  jupyter notebook image\_preprocessing.ipynb

&nbsp;  ```



3\. \*\*Generate embeddings\*\*

&nbsp;  ```bash

&nbsp;  jupyter notebook cLip\_img\_embeddings.ipynb

&nbsp;  ```



4\. \*\*Evaluate zero-shot classification\*\*

&nbsp;  ```bash

&nbsp;  jupyter notebook Object\_Counting\_Zero\_Shot\_CLIP.ipynb

&nbsp;  ```



5\. \*(Optional)\* Fine-tune the EfficientNet model

&nbsp;  ```bash

&nbsp;  python CLIP\_Efficient\_Net\_Train.py

&nbsp;  ```



---



\## 📈 Example Results  

| Model | Top-1 Accuracy | MAE | Description |

|-------|----------------|-----|--------------|

| CLIP (Zero-Shot) | 78 % | 1.3 | Uses natural-language prompts, no training |

| EfficientNet-B0 (Fine-Tuned) | 93 % | 0.5 | Trained for 10 epochs on labeled data |



---



\## 🔮 Future Work  

\- Add object-detection to locate items within bins.  

\- Experiment with ViT or Swin-Transformer backbones.  

\- Deploy inference as a REST API or Streamlit demo.  



---



\## 👩‍💻 Author  

\*\*Bhoomika\*\*  

Built as a personal deep-learning project to explore vision-language models and warehouse automation.



---



\## 📜 License  

MIT License — free for research and educational use.  



