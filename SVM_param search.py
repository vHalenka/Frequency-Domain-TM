import subprocess
import numpy as np

def run_experiment_with_params(ensembles, epochs, resolution, binarized, batch_size, block_size, quant_multiple, quant_matrix, dct_step, output_dir):
    # Build the command for the SVM experiment
    command = [
        'python3', 'SVM.py',  # Assuming the SVM script is named 'svm_experiment.py'
        '--ensembles', str(ensembles),
        '--epochs', str(epochs),
        '--resolution', str(resolution),
        '--binarized', str(binarized),
        '--batch_size', str(batch_size),
        '--block_size', str(block_size),
        '--quant_multiple', str(quant_multiple),
        '--quant_matrix', quant_matrix,
        '--dct_step', str(dct_step),
        '--output_dir', output_dir
    ]
    
    # Use subprocess.Popen to run the command and print the output in real-time
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Print stdout and stderr in real-time
    for stdout_line in process.stdout:
        print(stdout_line, end='')  # Print the output line by line
    
    # Wait for the process to finish and get the return code
    process.stdout.close()
    returncode = process.wait()
    
    # Capture any remaining stderr (if needed)
    stderr_output = process.stderr.read()
    if stderr_output:
        print(stderr_output)

    return returncode


def main():
    # Define the parameters to test
    ensembles = 1
    epochs = 25
    resolution = 12
    binarized = True
    batch_size = 500
    block_size = 8
    quant_multiple = 1.0
    quant_matrix = "img_quant_matrix"
    dct_steps = [8, 5, 3]

    # Iterate over the parameter combinations
    for dct_step in dct_steps:
        output_dir = f"results_svm_{resolution}_{binarized}_{quant_multiple}_{dct_step}"
        print(f"Running SVM experiment with resolution={resolution}, binarized={binarized}, quant_matrix={quant_matrix}, block_size={block_size}, dct_steps={dct_step}")
        run_experiment_with_params(
            ensembles, epochs, resolution, binarized, batch_size, block_size, quant_multiple, quant_matrix, dct_step, output_dir
        )

if __name__ == "__main__":
    main()
