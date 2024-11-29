#!/bin/bash

# Cross-platform script to process datasets

# Function to check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Python 3 is not installed or not added to PATH. Please install it and try again."
        exit 1
    fi
}

# Function to check the operating system
detect_os() {
    OS_TYPE=$(uname)
    if [[ "$OS_TYPE" == "Linux" || "$OS_TYPE" == "Darwin" ]]; then
        echo "Detected Linux/MacOS environment."
        PYTHON_CMD="python3"
    elif [[ "$OS_TYPE" =~ "MINGW" || "$OS_TYPE" =~ "CYGWIN" ]]; then
        echo "Detected Windows environment."
        PYTHON_CMD="python"
    else
        echo "Unsupported operating system: $OS_TYPE"
        exit 1
    fi
}

# Function to run a Python script
run_python_script() {
    local script_name=$1
    echo "Running $script_name..."
    $PYTHON_CMD $script_name
    if [ $? -eq 0 ]; then
        echo "$script_name executed successfully."
    else
        echo "Error occurred while running $script_name. Exiting."
        exit 1
    fi
}

# Main workflow
main() {
    # Detect OS and set Python command
    detect_os

    if command -v pip &> /dev/null
    then
        echo "pip is installed, proceeding with installation."
    
    # Install dependencies from requirements.txt
        if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
        else
            echo "requirements.txt not found!"
        fi
    else
        echo "pip is not installed. Please install pip first."
    fi

    # Navigate to the Datasets directory
    echo "Navigating to the Datasets directory..."
    cd Datasets || { echo "Failed to navigate to Datasets directory. Exiting."; exit 1; }

    # Step 1: Download datasets
    run_python_script "downloadDatasets.py"

    # Step 2: Create parquet databases
    run_python_script "FromFullDatasetToSampledParquet.py"

    # Step 3: Delete the Parquet files directory
    echo "Deleting parquet file directory..."
    rm -rf parquet_files
    if [ $? -eq 0 ]; then
        echo "Parquet files directory deleted successfully."
    else
        echo "Failed to delete parquet files directory. Exiting."
        exit 1
    fi

    # Step 4: Navigate back to the parent directory
    echo "Returning to the parent directory..."
    cd .. || { echo "Failed to return to parent directory. Exiting."; exit 1; }

    echo "All steps completed successfully!"
}

# Run the main workflow
main
