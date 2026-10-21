# Handwritten Digits Explorer

**DATS 6401 – Visualization of Complex Data**  
**Week 8: Image Data Visualization**

## Project Overview

This project explores handwritten digit images using Python and Streamlit. It focuses on understanding image patterns, pixel intensities, and how machine learning models interpret image data.

## Dataset

The project uses the Handwritten Digits dataset from scikit-learn.

- 1,797 grayscale images
- 10 digit classes (0–9)
- Image size: 8 × 8 pixels
- Pixel intensity range: 0–16

## Features

- **Image Montage:** Displays handwritten digit samples to compare visual patterns.
- **Image Structure:** Explores pixel intensities and image characteristics.
- **Class Comparison:** Compares mean images and intensity distributions between two digit classes.
- **Saliency Map:** Uses pixel occlusion to visualize which pixels influence model predictions.
- **Stamp Experiment:** Demonstrates how an artificial feature can influence classification results.
- **Interactive Browser:** Allows users to explore images and visualizations through Streamlit.

## Key Findings

- The dataset is approximately balanced across digit classes.
- Handwritten digits show variations in shape and pixel intensity.
- Occlusion saliency reveals that some pixels influence predictions more than others.
- In the stamp experiment, digit 3 recall increased from 26.1% without the artificial stamp to 100% with it, suggesting shortcut learning.

## Tools and Libraries

- Python
- Streamlit
- NumPy
- Matplotlib
- scikit-learn

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Data Source

scikit-learn Handwritten Digits Dataset (`load_digits`).
