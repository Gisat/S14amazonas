import subprocess

def get_folder_size(remote_name, folder_path):
    try:
        command = f"rclone size {remote_name}{folder_path} --config /home/eouser/userdoc/rclone.conf"
        print(f"Executing command: {command}")  # Print the command to be executed
        result = subprocess.check_output(command, shell=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"

# Example usage
remote_name = "gisat:"  # Replace with your remote name
folder_path = "amazonas-archive/output/StatCubes/21LYG"  # Replace with your folder path

folder_size = get_folder_size(remote_name, folder_path)
print(folder_size)