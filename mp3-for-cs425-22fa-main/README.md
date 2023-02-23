# README for MP3

## How to run the project:

1. run FMaster.py at virtual machine 1.
     ```
    python3 FMaster.py
   ```
2. run file_server.py at virtual machine 1.
    ```
   python3 file_server.py
   ```
3. continue to join all the other node., by running file_server.py at them

## commands are listed as follow:
- put localfilename sdfsfilename
- get sdfsfilename localfilename
- delete sdfsfilename
- get_versions sdfsfilename num_versions localfilename
- ls sdfsfilename
- store