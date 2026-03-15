import zipfile
import os
import sys

def zip_folder(folder_path, output_path):
    print(f"Creating {output_path} from {folder_path}...")
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Count total files for progress reporting
    total_files = sum(len(files) for _, _, files in os.walk(folder_path))
        
    print(f"Found {total_files} files to compress. Starting incremental zip...")

    processed_files = 0
    # using zipfile directly allows us to write incrementally, minimizing RAM use.
    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, os.path.dirname(folder_path))
                    
                    zipf.write(file_path, rel_path)
                    
                    processed_files += 1
                    if processed_files % 100 == 0:
                        # Print progress on the same line
                        sys.stdout.write(f"\rProgress: {processed_files}/{total_files} files processed...")
                        sys.stdout.flush()
                        
        print(f"\nSuccessfully created {output_path}. Memory usage should have remained stable.")
    except Exception as e:
        print(f"\nError creating zip file: {e}")

if __name__ == '__main__':
    zip_folder('C-NMC_Dataset', 'C-NMC_Dataset.zip')
