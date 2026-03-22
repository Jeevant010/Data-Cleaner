# Supervised Audio Classification Trial

This notebook (`supervised_model_trial.ipynb`) is designed to train a Machine Learning model specifically tailored for small audio datasets (~100 to 500 samples). Rather than using data-hungry Deep Learning approaches, it extracts robust frequency features and trains a classic Random Forest Classifier to prevent overfitting.

## How the Pipeline Works
1. **Feature Extraction**: Uses the `librosa` library to extract MFCCs (Mel-Frequency Cepstral Coefficients). This converts raw `.wav` audio waves into a compact array of numbers representing the sound's "texture".
2. **Data & Labels**: Scans your `Data/Output` folder for the trimmed audio clips. You can easily adjust the logic to label files as "Gunshot" (1) or "Noise" (0) based on their filenames or subfolders.
3. **Training**: Splits the dataset into 80% training data and 20% testing data, then fits a `RandomForestClassifier`.
4. **Evaluation**: Generates a detailed Classification Report (showing Precision, Recall, and Accuracy) and neatly plots a visual Convolution Matrix heatmap so you know exactly what your model is misclassifying.

## How to Use the Notebook
1. **Setup Environment**: Ensure you run the first cell to install the necessary ML packages (`scikit-learn`, `librosa`, `pandas`, `seaborn`).
2. **Organize Your Data**: Place your trimmed `.wav` clips into the `dataset_dir` path (currently defaults to `c:\Desktop\Data-Cleaner\research\Data\Output`). 
   - *Tip:* The notebook currently looks for the string "noise" in the filename to automatically assign a label of 0. If you instead organize them into separate folders, simply tweak the `labels.append()` logic block!
3. **Run All Cells**: Execute the notebook cells from top to bottom in order.
4. **Review Results**: Scroll to the very bottom to see the model's accuracy on the 20% test holdout set!
