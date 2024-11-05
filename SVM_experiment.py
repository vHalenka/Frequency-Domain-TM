import numpy as np
import cv2
from keras.datasets import mnist
from sklearn import svm
from tqdm import tqdm
import os
import time
import logging
import argparse
import wandb
from skimage.util import view_as_windows
from sklearn.metrics import accuracy_score
from scipy.stats import boxcox  # Import Box-Cox function

# Set logging level for matplotlib to WARNING or ERROR
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Quantization matrices remain unchanged
img_quant_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

text_quantization_matrix = np.array([
    [8, 6, 5, 8, 12, 20, 24, 30],
    [6, 6, 7, 10, 14, 26, 28, 25],
    [7, 7, 8, 12, 20, 27, 35, 29],
    [7, 9, 11, 14, 24, 40, 36, 28],
    [9, 11, 19, 29, 34, 55, 52, 39],
    [12, 18, 28, 32, 40, 52, 57, 46],
    [25, 32, 39, 40, 52, 61, 60, 51],
    [36, 46, 48, 49, 56, 50, 52, 50]
])

def quantize(X_dct, quant_matrix):# 4,4 8,8
    h, w = X_dct.shape[-2:]
    quantized_dct = np.zeros_like(X_dct)

    max_qh = min(h,quant_matrix.shape[0])
    max_qw = min(w,quant_matrix.shape[1])
    for row in range(X_dct.shape[0]):
        for col in range(X_dct.shape[1]):
            for i in range(0, h, max_qh):
                for j in range(0, w, max_qw):
                    block = X_dct[row, col ,i:i+max_qh, j:j+max_qw]
                    # Get the actual size of the block (handle edge cases where block size is smaller)
                    block_h, block_w = block.shape
                    
                    # Use only the relevant part of the quantization matrix
                    quant_matrix_block = quant_matrix[...,:block_h, :block_w]
                    
                    # Quantize the block
                    quantized_dct[...,i:i+8, j:j+8] = np.round(block / quant_matrix_block) * quant_matrix_block
    
    return quantized_dct

def extract_dct_from_image(img, block_size=8, step=8, tile=False, crop=False):
    h, w = img.shape    
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    image_padded = np.pad(np.float32(img), ((0, h_pad), (0, w_pad)), 'constant', constant_values=0)
    
    blocks = view_as_windows(image_padded, (block_size, block_size), step)
    dct_blocks = np.zeros_like(blocks, dtype=np.float32)
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            block = blocks[i, j]
            dct_blocks[i, j] = cv2.dct(block)
    
    if tile == True: # Return to a single image?
        dct_image = np.block([[dct_blocks[i, j] for j in range(dct_blocks.shape[1])] 
                                            for i in range(dct_blocks.shape[0])])
    else:
        #dct_image = dct_blocks.reshape(dct_blocks.shape[0], dct_blocks.shape[1], -1)
        dct_image = dct_blocks
    if crop == True: # Set to original size?
        return dct_image[:h, :w]
    else:
        return dct_image


def apply_boxcox_to_dataset(X_data):
    # Flatten the dataset while keeping the samples dimension
    X_flattened = X_data.reshape(X_data.shape[0], -1)
    
    # Apply Box-Cox transformation to the entire dataset globally
    # Ensure all values are positive for Box-Cox
    X_flattened = X_flattened + np.abs(np.min(X_flattened)) + 1e-10
    
    # Apply Box-Cox to each feature (column-wise)
    X_boxcox = np.zeros_like(X_flattened)
    for i in range(X_flattened.shape[1]):
        X_boxcox[:, i], _ = boxcox(X_flattened[:, i])
    
    # Reshape back to the original dimensions (e.g., 28x28 for images)
    return X_boxcox.reshape(X_data.shape)

def thermometer_encoding(X_data, resolution, clip_percentile=99):
    X_min = np.min(X_data)
    X_max = np.max(X_data)
    
    lower_bound = np.percentile(X_data, 100 - clip_percentile)
    upper_bound = np.percentile(X_data, clip_percentile)
    
    X_clipped = np.clip(X_data, lower_bound, upper_bound)
    X_scaled = (X_clipped - X_min) / (X_max - X_min)
    
    X_encoded = np.zeros((*X_scaled.shape, resolution), dtype=np.uint32)
    for z in range(resolution):
        threshold = (z + 1) / (resolution + 1)
        X_encoded[... , z] = X_scaled >= threshold  
    
    return X_encoded

def add_noise(X_in, bits):
    noise_shape = list(X_in.shape)
    noise_shape[-1] = bits
    X_noise = np.random.randint(0, 2, size=noise_shape, dtype=np.uint32)
    with_noise = np.concatenate((X_in, X_noise), axis=-1)
    return with_noise

def binarize(X_batch):
    # Convert to 8 bit
    X_batch_8bit = np.array([
        cv2.convertScaleAbs(
            img.astype(np.uint8),
            alpha=(255.0 / np.max(img) if np.max(img) != 0 else 1.0)
        )
        for img in X_batch
    ])
    
    # Single Channel (CIFAR10 adapt)
    X_batch_gray = np.array([
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        for img in X_batch_8bit
    ])
    
    # Binarize using adaptiveThreshold
    X_batch_binarized = np.array([
        cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        for img in X_batch_gray
    ])
    X_batch_binarized = np.squeeze(X_batch_binarized)
    return X_batch_binarized

def preprocess_data_in_batches(X_data, block_size, resolution, batch_size, binarised, quant_multiple=1, quant_matrix="img_quant_matrix", dct_step=8):
    num_samples = X_data.shape[0]
    noise_bits = 0
    X_processed_batches = []

    # Map the quant_matrix argument to the actual matrix
    if quant_matrix == "img_quant_matrix":
        quant_matrix = img_quant_matrix
    elif quant_matrix == "text_quantization_matrix":
        quant_matrix = text_quantization_matrix
    else:
        raise ValueError(f"Unknown quantization matrix: {quant_matrix}")

    # Loop over the data in batches
    for i in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
        X_batch = X_data[i:i + batch_size]
        for img_idx, img_tot in enumerate(X_batch):
            if binarised:
                img_tot = binarize(img_tot)
            
            img_tot = extract_dct_from_image(img_tot, block_size=block_size, step=dct_step)            
            img_tot = quantize(img_tot, quant_matrix * quant_multiple)
            img_tot = img_tot.astype(np.uint32)
            #img_tot = img_tot.transpose(0, 2, 1, 3).reshape(img_tot.shape[0] * img_tot.shape[2], img_tot.shape[1] * img_tot.shape[3])            
            #img_tot = thermometer_encoding(img_tot, resolution)
            #img_tot = add_noise(img_tot, bits=noise_bits)
            
            # Append each processed image to the list
            X_processed_batches.append(img_tot)
    
    # Convert the list of processed images into a NumPy array
    X_processed_batches = np.array(X_processed_batches, dtype=np.uint32)
    
    print("Preprocessed into shape: ", X_processed_batches.shape)
    return X_processed_batches


def extract_dct_from_array(image_array, block_size):
    h, w = image_array.shape
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - h % block_size) % block_size
    image_padded = np.pad(image_array, ((0, h_pad), (0, w_pad)), 'constant', constant_values=0)

    dct_array = np.zeros_like(image_padded, dtype=np.float32)
    for i in range(0, image_padded.shape[0], block_size):
        for j in range(0, image_padded.shape[1], block_size):
            block = image_padded[i:i + block_size, j:j + block_size]
            dct_array[i:i + block_size, j:j + block_size] = cv2.dct(np.float32(block))
    
    return dct_array[:h, :w]

# Main function to run the experiment
def run_experiment(X_train, Y_train, X_test, Y_test, ensembles, epochs, resolution, output_base_dir, batch_size, binarized, block_size, quant_multiple, quant_matrix=img_quant_matrix, dct_step=8):
    num_samples = X_train.shape[0]
    
    # Initialize a new WandB run and log the config parameters
    wandb.init(project="DCT_SVM")
    
    # Log all the passed arguments (config) into WandB
    wandb.config.update({
        "ensembles": ensembles,
        "epochs": epochs,
        "resolution": resolution,
        "binarized": binarized,
        "batch_size": batch_size,
        "block_size": block_size,
        "quant_multiple": quant_multiple,
        "quant_matrix": quant_matrix,
        "output_base_dir": output_base_dir,
        "dct_step": dct_step
    })
    # Preprocess the data in batches before training
 
    output_dir = os.path.join(output_base_dir, f"bin_{binarized}, resolution_{resolution}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ensemble in range(ensembles):
        # Initialize an SVM model
        model = svm.SVC(kernel='linear')  # You can adjust kernel and other SVM parameters
        
        print("Preprocessing test data...")
        X_test_processed = preprocess_data_in_batches(X_test, block_size=block_size, resolution=resolution, batch_size=batch_size, binarised=binarized, quant_multiple=quant_multiple, quant_matrix=quant_matrix, dct_step=dct_step)
        X_test_processed = X_test_processed.reshape(X_test_processed.shape[0], -1)  # Flatten data for SVM

        # Parameters for early stopping
        patience = 3  # Number of epochs with no improvement after which training will be stopped
        min_delta = 0.1  # Minimum change to qualify as an improvement
        best_accuracy = 0
        epochs_no_improve = 0

        for epoch in range(epochs):
            start_time = time.time()
            for i in tqdm(range(0, num_samples, batch_size), desc="Training batches"):

                X_train_batch = X_train[i:i + batch_size]
                Y_train_batch = Y_train[i:i + batch_size].astype(np.uint32)

                print("Preprocessing training data...")
                X_train_batch_processed = preprocess_data_in_batches(X_train_batch, block_size=block_size, resolution=resolution, batch_size=batch_size, binarised=binarized, quant_multiple=quant_multiple, quant_matrix=quant_matrix, dct_step=dct_step)

                X_train_batch_processed = X_train_batch_processed.reshape(X_train_batch_processed.shape[0], -1)  # Flatten data for SVM
                
                model.fit(X_train_batch_processed, Y_train_batch)

            X_test_subset = X_test_processed
            Y_test_subset = Y_test[:].astype(np.uint32)

            predictions = model.predict(X_test_subset)
            result_test = 100 * accuracy_score(Y_test_subset, predictions)
            epoch_time = time.time() - start_time
            # Check for improvement in accuracy
            if result_test - best_accuracy > min_delta:
                best_accuracy = result_test
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Stopping early at epoch {epoch+1}. No improvement for {patience} consecutive epochs.")
                break
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
            
            # Log accuracy and training time per epoch to WandB
            wandb.log({
                "epoch": epoch + 1,
                "test_accuracy": result_test,
                "epoch_time": epoch_time,
                "X_train_batch_shape": X_train_batch.shape
            })               

            print(f"Epoch {epoch + 1}, Resolution {resolution}: Test Accuracy {result_test:.2f}%, Training Time: {epoch_time:.2f} seconds")
    wandb.finish()
    return result_test


def main():
    parser = argparse.ArgumentParser(description="Run SVM experiment with MNIST")
    
    # Command line arguments (removed TM-specific ones)
    parser.add_argument('--ensembles', type=int, default=1, help='Number of ensembles')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--resolution', type=int, default=20, help='Resolution for thermometer encoding')
    parser.add_argument('--binarized', type=bool, default=True, help='Use binarization')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size')
    parser.add_argument('--block_size', type=int, default=8, help='Block size for DCT')
    parser.add_argument('--quant_multiple', type=float, default=1.0, help='Multiplies Quantization table')
    parser.add_argument('--quant_matrix', type=str, default="img_quant_matrix", help='Quantization table selection')
    parser.add_argument('--dct_step', type=int, default=8, help='DCT window skip pixels')
    parser.add_argument('--output_dir', type=str, default="results", help='Output directory for results')
    
    args = parser.parse_args()

    # Load MNIST dataset
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)

    # Take a subset for the purpose of this study
    samples = 1000
    X_train = X_train[:samples].astype(np.uint32)
    Y_train = Y_train[:samples].astype(np.uint32)

    samples2 = 100
    X_test = X_test[:samples2].astype(np.uint32)
    Y_test = Y_test[:samples2].astype(np.uint32)

    X_train = X_train.astype(np.uint32)
    X_test = X_test.astype(np.uint32)

    # Run the experiment with the provided command-line parameters
    run_experiment(X_train, Y_train, X_test, Y_test,
                ensembles=args.ensembles,
                epochs=args.epochs,
                resolution=args.resolution,
                output_base_dir=args.output_dir,
                batch_size=args.batch_size,
                binarized=args.binarized,
                block_size=args.block_size,
                quant_multiple=args.quant_multiple,
                quant_matrix=args.quant_matrix,
                dct_step=args.dct_step)

    # Finish WandB logging
    wandb.finish()

if __name__ == "__main__":
    main()
