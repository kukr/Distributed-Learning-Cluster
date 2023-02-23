# README for MP3

## Please install the following required python libraries

pip3 install pillow==5 <br />
pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

## How to run the project:

1. run FMaster.py at virtual machine 1.
     ```
   python3 FMaster.py
   ```
2. run FStandby.py at virtual machine 2 or the hot standby of your chosen.
    ```
   python3 file_server.py
   ```
2. run file_server.py at virtual machine 1.
    ```
   python3 file_server.py
   ```
3. continue to join all the other node., by running file_server.py at them

Start the models in inference phase by using the following commands
```
   infer alexnet
   infer resnet
```

This will load the pretrained models and get them ready for inference phase

## commands are listed as follow:

- print_statistics - Prints all the statistics such as the average_query_rate, median, std.dev, 90th percentile, 95th percentile, 99th percentile
- total_queries - Prints the total_queries that have been processed so far
- query_info - Prints the query rates of the jobs that are assigned
- vm_job_dict - Prints what models are assigned to what VMs
