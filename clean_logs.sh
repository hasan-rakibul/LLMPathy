#!/bin/bash

# Function to clean log files of a specific pattern
clean_files() {
    local dir=$1
    local pattern=$2
    echo "Searching for files matching pattern '$pattern' in directory '$dir'..."
    
    # Find matching files
    files=$(find "$dir" -type f -name "$pattern")
    if [[ -z "$files" ]]; then
        echo "No files found matching '$pattern'."
        return
    fi

    # Display files and ask for confirmation
    echo "The following files will be deleted:"
    echo "$files"
    read -p "Do you want to delete these files? (y/n): " choice
    if [[ "$choice" == "y" ]]; then
        find "$dir" -type f -name "$pattern" -exec rm -f {} +
        echo "Files deleted."
    else
        echo "Skipping deletion of '$pattern' files."
    fi
}

# Main script
echo "Enter the parent directory:"
read parent_dir

# Check if directory exists
if [[ ! -d "$parent_dir" ]]; then
    echo "Error: Directory '$parent_dir' does not exist."
    exit 1
fi

# Step-by-step cleaning
clean_files "$parent_dir" "*.ckpt"
clean_files "$parent_dir" "events.out.tfevents.*"

echo "Cleaning process completed."
