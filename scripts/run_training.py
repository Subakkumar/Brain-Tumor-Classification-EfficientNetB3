import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from brain_tumor_classifier import BrainTumorClassifier

def main():
    DATA_PATH = r"D:\Suba Projects\P1-Brain-Tumor-Classification-EfficientNetB3\DataSet Brain Tumor"
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at: {DATA_PATH}")
        return
    
    print("🚀 Starting Brain Tumor Classification...")
    classifier = BrainTumorClassifier(data_dir=DATA_PATH)
    classifier.run_complete_pipeline()

if __name__ == "__main__":
    main()