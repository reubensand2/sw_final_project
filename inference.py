import numpy as np
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import os

# --- 1. The Hardware-Equivalent Class ---
class SNN_Hardware_GoldenModel:
    def __init__(self, beta=0.95, threshold=1.0):
        self.beta = beta
        self.threshold = threshold
        
        # Load Weights & Biases
        self.w1 = np.load("weights_fc1.npy")         # Shape: (128, 128)
        self.b1 = np.load("bias_fc1.npy")         # Shape: (128,)
        self.w2 = np.load("weights_fc2.npy")         # Shape: (128, 6)
        self.b2 = np.load("bias_fc2.npy")         # Shape: (6,)
        
        # Load Normalization Constants
        self.mean = np.load("scaler_mean.npy")
        self.scale = np.load("scaler_scale.npy")

    def normalize(self, raw_input):
        """
        Hardware Pre-processing Block.
        Eq: (Input - Mean) / Scale
        """
        # Epsilon prevents division by zero if scale is 0 (unlikely here but good practice)
        return (raw_input - self.mean) / (self.scale + 1e-10)

    def forward(self, x_raw, num_steps=20):
        """
        Runs the SNN for one single sample (row of data).
        """
        # 1. Pre-processing
        x = self.normalize(x_raw)
        
        # 2. Initialize Neuron States (Membrane Potential)
        # In HLS, these are registers initialized to 0
        mem1 = np.zeros(self.w1.shape[1]) # 128 hidden neurons
        mem2 = np.zeros(self.w2.shape[1]) # 6 output neurons
        
        output_spikes_counter = np.zeros(6)
        
        # 3. Pre-calculate Constant Inputs (Direct Encoding optimization)
        # Since Input X is static, the current into Layer 1 is constant for all time steps.
        # We calculate it once to save 20x matrix multiplications.
        # Current = (Input * Weights) + Bias
        input_current_1 = np.dot(x, self.w1) + self.b1
        
        # --- TIME LOOP (The "Tick") ---
        for t in range(num_steps):
            
            # === LAYER 1 (Hidden) ===
            # Lif Logic: V[t] = V[t-1]*beta + Input_Current
            mem1 = (mem1 * self.beta) + input_current_1
            
            # Spike Logic: If V > Thr, Spike = 1, else 0
            spikes1 = (mem1 > self.threshold).astype(int)
            
            # Reset Logic (Subtractive): V = V - (Spike * Thr)
            mem1 = mem1 - (spikes1 * self.threshold)
            
            # === LAYER 2 (Output) ===
            # Input to L2 changes every step because spikes1 changes!
            # Current = (Spikes1 * Weights2) + Bias2
            input_current_2 = np.dot(spikes1, self.w2) + self.b2
            
            # Lif Logic
            mem2 = (mem2 * self.beta) + input_current_2
            
            # Spike Logic
            spikes2 = (mem2 > self.threshold).astype(int)
            
            # Reset Logic
            mem2 = mem2 - (spikes2 * self.threshold)
            
            # Accumulate Output for Classification
            output_spikes_counter += spikes2
            
        return output_spikes_counter

# --- VISUALIZATION ROUTINE ---
def run_drift_analysis():
    # List of all batch files
    # Adjust path './Dataset/' to match your folder structure exactly
    batch_files = [f'./Dataset/batch{i}.dat' for i in range(1, 11)]
    
    accuracies = []
    batch_indices = []

    model = SNN_Hardware_GoldenModel(beta=0.95)

    print(f"{'Batch':<10} | {'Samples':<10} | {'Accuracy':<10}")
    print("-" * 35)

    for i, file_path in enumerate(batch_files):
        if not os.path.exists(file_path):
            print(f"Batch {i+1}: File not found ({file_path})")
            continue
            
        # 1. Load Data
        X_sparse, y = load_svmlight_file(file_path, n_features=128)
        X_data = X_sparse.toarray()
        y_data = y - 1 
        
        # 2. Run Inference
        correct = 0
        total = len(y_data)
        
        for j in range(total):
            spike_counts = model.forward(X_data[j], num_steps=20)
            prediction = np.argmax(spike_counts)
            if prediction == y_data[j]:
                correct += 1
        
        # 3. Record Stats
        acc = 100 * correct / total
        accuracies.append(acc)
        batch_indices.append(i + 1)
        
        print(f"Batch {i+1:<4} | {total:<10} | {acc:.2f}%")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.plot(batch_indices, accuracies, marker='o', linestyle='-', linewidth=2, color='b')
    
    # Highlight Training Batches (Assuming you trained on Batch 1 & 2)
    plt.axvspan(1, 2.5, color='green', alpha=0.1, label='Training Data')
    
    plt.title('SNN Accuracy over Time (Gas Sensor Drift)')
    plt.xlabel('Batch Number (Time)')
    plt.ylabel('Accuracy (%)')
    plt.xticks(batch_indices)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    run_drift_analysis()