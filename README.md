# Google Lens Alternative: Image Similarity Search

This project is an implementation of an alternative to Google Lens' feature image similarity search system using multiple approaches. The objective is to explore and compare distinct methods for image similarity retrieval, evaluating their performance on metrics, computational efficiency and scalability for real-time usage scenarios.


Image similarity search works by extracting embeddings/features from images and comparing them in a high-dimensional feature space. The closer the features of two images are in this space, the more similar they are considered to be. Process leverages deep learning models and distance metrics like cosine similarity or Euclidean distance to find similarity.

## **How to Run**
All the approaches are consolidated in `combined_approaches.ipynb`, which can be executed directly. I've used Google Colab for my experiments.


## **Implemented Approaches**
1. **ResNet50**:
   - Pre-trained Resnet50 convolutional neural network used to extract embeddings from images.
   - Can use transfer learning to enhance performance on the dataset.
2. **Autoencoder**:
   - Designed to learn a compact latent space representation of images.
   - Fine-tuned for unsupervised feature extraction and similarity search on Fashion MNIST dataset.
3. **Siamese Network**:
   - Trained to measure similarity between image pairs by learning a contrastive loss function.
   - Suitable for applications requiring pairwise comparisons.
4. **CLIP (Contrastive Language-Image Pretraining)**:
   - Pre-trained model by OpenAI that aligns images in a shared embedding space.

## **Dataset**
Since no dataset was provided, I have used Fashion MNIST dataset for my experiments.

## **Evaluation Metrics**
The following metrics were used to evaluate the performance of each approach:
- **Precision**: Proportion of relevant images retrieved out of the total retrieved.
- **Recall**: Proportion of relevant images retrieved out of the total relevant images in the dataset.
- **Retrieval Accuracy**: Overall accuracy of retrieved results matching ground truth. F1 Score

Note: Precision and Recall is similar here, as we don't have any fixed number of images in dataset to search for. Ifwe consider only fashion mnist or any other dataset, then recall would be different.

# Use Precision here

In real-time systems (like Google Lens), precision may be prioritized to ensure a seamless user experience.

Example: In an e-commerce platform, if a user searches for "red dresses," and the system retrieves 10 images of which 8 are truly red dresses, the precision is 80%. A higher precision ensures that users see mostly relevant results, improving their experience.


# Results 

| **Approach**               | **Evaluation Metrics (Precision, Recall, Retrieval Accuracy)**  | **Computational Efficiency**          | **Scalability & Infrastruce Required**                                        |  **Comment** |
|-----------------------------|---------------|-------------------------|--------------------------------|-----------------------------------------------------|
| ResNet50               | High          | Very High               | High  with low-end GPUs | Good with low range dataset. Bad in capturing gloabl context. No need for training, use finetune on one dataset will work or without finetuning as well          | 
| Autoencoder                   | Moderate           | Moderate                | High with medium-end GPUs | Needs proper training on dataset. Captures global and local features effectively.     |
| Siamese Network                       | Moderate                   | Low                | High with medium-end GPUs | Needs proper training on dataset.                              |
| CLIP           | Very High     | Moderate         | High  with high-end GPUs  | Very good in almost any type of data. Non need for training or fine tuning   |


# Insights

## CLIP
- **Best for general-purpose image similarity tasks across diverse datasets.**  
- No fine-tuning required, making it suitable for real-time usage in production.  
- However, its higher computational cost may be a constraint for resource-limited applications.  
- Ideal for tasks requiring a balance of scalability and robustness across different and wiide domains.  

## Siamese Networks
- Designed for tasks requiring pairwise comparisons, such as verifying similarity between two specific images.  
- Training on smaller datasets leads to effective results but struggles with scalability when applied to large datasets.  
- Recommended for niche use cases requiring high accuracy in controlled environments.  

## ResNet50
- Offers high computational efficiency and does not require extensive fine-tuning for small-scale datasets.  
- Performs well for retrieving similar images when global context is less critical.  
- Suggested for applications with resource constraints or when working with a low-range dataset.  

## Autoencoder
- Lightweight and fast, making it suitable for edge devices or tasks with limited computational resources.  
- Captures both local and global features but requires proper training on the target dataset.  
- A good option for scenarios prioritizing speed and simplicity over precision.  


# My Recommendation
Based on the evaluation and assignment requirements, I recommend CLIP for real-time image similarity search:

Recommended Model: CLIP

**Why?** - Because it offers the best balance between scalability and performance, with high retrieval accuracy across diverse datasets. Its ability to work without retraining or fine-tuning makes it ideal for dynamic, real-world applications where new data is frequently added.

Ideal for systems like Google Lens or large-scale e-commerce platforms with varied and continuously changing image catalogs.


## **References**
- Deep Residual Learning for Image Recognition. arXiv:1512.03385
- Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks.
- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks for One-shot Image Recognition. https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
- Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020

