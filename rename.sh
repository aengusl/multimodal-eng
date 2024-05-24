#!/bin/bash

# Define the directory containing the files
directory="/root/multimodal-eng/pics_of_mates_faces"

# Function to rename files in the directory
rename_files_in_directory() {
    local dir="$1"
    for filepath in "$dir"/*; do
        # Check if it is a directory
        if [[ -d "$filepath" ]]; then
            # Recursively rename files in subdirectory
            rename_files_in_directory "$filepath"
        else
            # Get the base name of the file
            base_name=$(basename "$filepath")
            
            # Replace non-compliant characters with underscores
            new_base_name=$(echo "$base_name" | sed 's/[^a-zA-Z0-9_.]/_/g')
            
            # Generate a unique identifier to avoid name collisions
            unique_id=$(uuidgen)
            
            # Add the unique identifier to the filename before the extension
            name="${new_base_name%.*}"
            ext="${new_base_name##*.}"
            new_filename="${name}_${unique_id}.${ext}"
            
            # Rename the file
            new_filepath="$dir/$new_filename"
            mv "$filepath" "$new_filepath"
            
            echo "Renamed '$base_name' to '$new_base_name'"
        fi
    done
}

# Rename the files in the directory
rename_files_in_directory "$directory"