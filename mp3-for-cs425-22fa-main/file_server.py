import server
from server import encode_command, decode_command, encode_ping_ack, decode_ping_ack
import threading
import socket
import os
import time
import struct
import json
from torchvision import models
from torchvision import transforms
import torch
from PIL import Image
from collections import defaultdict
import numpy as np


BUFFER_SIZE = 4096
MASTER_HOST = INTRODUCER_HOST = socket.gethostbyname('fa22-cs425-5402.cs.illinois.edu')
HOTSTANDY_HOST = socket.gethostbyname('fa22-cs425-5403.cs.illinois.edu')
MACHINE_NUM = int(socket.gethostname()[13:15])
LOG_FILEPATH = f'machine.{MACHINE_NUM}.log'
PING_PORT = 20240
MEMBERSHIP_PORT = 20241
PING_INTERVAL = 2.5
PING_TIMEOUT = 2
MASTER_PORT = 20086
FILE_PORT = 10086
GET_ADDR_PORT = 10087
APPEND_PORT = 50054
INFER_PORT = 50056
MODEL_INIT_PORT = 50058
QUERY_RATE_PORT = 50062
BATCH_ACK_PORT = 50064
GET_QUERY_INFO_PORT = 50065
GET_VM_JOB_DICT_PORT = 50066
MODEL_OUT_PORT = 50059
SCHEDULE_JOB_PORT = 50060
STATISTICS_PORT = 50070
TOTAL_QUERIES_PORT = 50071
#SET_BATCH_SIZE_PORT = 50067
ALL_VMS = ['172.22.156.179','172.22.158.179','172.22.158.181']#,'','','','','','','']
DATASET_PATH = '/home/uk3/mp3-for-cs425-22fa-main/dataset/ILSVRC_images/'

def send_file(conn: socket.socket, localfilepath, sdfsfileid, timestamp):
    header_dic = {
        'sdfsfileid': sdfsfileid,  # 1.txt
        'timestamp': timestamp,
        'file_size': os.path.getsize(localfilepath)
    }
    header_json = json.dumps(header_dic)
    header_bytes = header_json.encode()
    conn.send(struct.pack('i', len(header_bytes)))
    conn.send(header_bytes)
    with open(localfilepath, 'rb') as f:
        for line in f:
            conn.send(line)

def receive_file(conn: socket.socket):
    obj = conn.recv(4)
    header_size = struct.unpack('i', obj)[0]
    header_bytes = conn.recv(header_size)
    header_json = header_bytes.decode()
    header_dic = json.loads(header_json)
    total_size, sdfsfileid, timestamp = header_dic['file_size'], header_dic['sdfsfileid'], header_dic['timestamp']

    data = b''
    recv_size = 0
    while recv_size < total_size:
        line = conn.recv(BUFFER_SIZE)
        data += line
        recv_size += len(line)
    return data, sdfsfileid, timestamp

class FServer(server.Node):
    class FileTable:
        def __init__(self):
            self.file_lookup = {}
            self.file_lookup_lock = threading.Lock()

        def _insert(self, arr, e):
            l, r = 0, len(arr)
            while l<r:
                mid=(l+r)//2
                if arr[mid] >= e:
                    r = mid
                else:
                    l = mid + 1
            arr.insert(l, e)

        def insert_file(self, file, sdfsfileid, timestamp):
            self.file_lookup_lock.acquire()
            self.file_lookup.setdefault(sdfsfileid, [])
            self._insert(self.file_lookup[sdfsfileid], timestamp)
            self.file_lookup_lock.release()
            sdfsfilename = sdfsfileid + '-' + str(timestamp)
            with open(sdfsfilename, 'wb') as f:
                f.write(file)

        def delete_file(self, sdfsfileid):
            self.file_lookup_lock.acquire()
            self.file_lookup.pop(sdfsfileid)
            self.file_lookup_lock.release()
            # delete file

        def check_file(self, sdfsfileid):
            if sdfsfileid not in self.file_lookup:
                return None
            return self.file_lookup[sdfsfileid][-1]

        def show_file(self):
            print('files stored at this machine:')
            self.file_lookup_lock.acquire()
            for i in self.file_lookup.keys():
                print(' ', i)
            self.file_lookup_lock.release()

        def get_n_versions(self, sdfsfileid, n):
            self.file_lookup_lock.acquire()
            n = min(n, len(self.file_lookup[sdfsfileid]))
            timestamps = [i for i in self.file_lookup[sdfsfileid][-n:]]
            self.file_lookup_lock.release()

            data = []
            for t in timestamps:
                with open(sdfsfileid+'-'+str(t), 'rb') as f:
                    data.append(f.read())
            return data

    def ping_thread(self):
        # generate ping_id
        ping_id = self.id + '-' + str(self.ping_count)
        self.ping_count += 1
        # encode ping
        encoded_ping = encode_ping_ack(ping_id, {'type': 'ping', 'member_id': self.id})
        # initialize cache for the ping_id
        self.ack_cache_lock.acquire()
        self.ack_cache[ping_id] = set()
        self.ack_cache_lock.release()
        # transmit ping, get the ids of the member that's been pinged
        for i in range(4):
            ids = self.transmit_message(encoded_ping, self.ping_port)
        # wait for some time to receive ack
        time.sleep(self.ping_timeout)
        # get the received ack
        self.ack_cache_lock.acquire()
        ack_cache_for_this_ping_id = self.ack_cache[ping_id]
        self.ack_cache.pop(ping_id)
        self.ack_cache_lock.release()
        # check all the acks that's received
        fail_ip = []
        for id in ids:
            if id not in ack_cache_for_this_ping_id: # if an ack is not received
                fail_ip.append(id.split(':')[0])
                new_membership_list = self.update_membership_list(0, id) # get updated membership_list by deleting the member that's missing

                # assign unique command id
                self.command_lock.acquire()
                command_id = self.id + '-' + str(self.command_count)
                self.command_count += 1
                self.commands.add(command_id)
                self.command_lock.release()

                # encode command
                command_content = {'type': 'failed', 'content': id}
                encoded_command_tosend = encode_command(command_id, command_content)
                self.mode_lock.acquire()
                if self.debug:
                    self.mode_lock.release()
                    print("haven't receiving ack from ", id)
                    print('sending command ', command_content) # print statement for debugging
                else:
                    self.mode_lock.release()

                # transmit message, using old membership_list
                self.transmit_message(encoded_command_tosend, self.membership_port)

                # update membership list
                self.membership_lock.acquire()
                self.membership_list = new_membership_list
                self.membership_lock.release()
                self.log_generate(id, 'failed', self.membership_list)
        if fail_ip:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(json.dumps({'command_type': 'fail_notice', 'command_content': fail_ip}).encode(), (self.master_ip, self.master_port))


    def __init__(self, ping_port: int, membership_port: int, ping_timeout: int, ping_interval: int, log_filepath: str, file_port: int, master_port: int, master_host: str):
        super().__init__(ping_port, membership_port, ping_timeout, ping_interval, log_filepath)
        self.file_port = file_port
        self.file_cache = {}
        self.file_lock = threading.Lock()
        self.file_table = self.FileTable()
        self.put_lock = threading.Lock()
        self.get_lock = threading.Lock()
        self.put_ack_cache = {}
        self.get_ack_cache = {}
        self.ls_cache = {}
        self.ls_lock = threading.Lock()
        self.master_port = master_port
        self.master_ip = socket.gethostbyname(master_host)

    def get_ip(self, sdfsfileid):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.master_ip, GET_ADDR_PORT))
            except socket.error as e:
                return
            s.send(sdfsfileid.encode())
            ips = json.loads(s.recv(4096).decode())
            return ips

    # utilities
    def filehash(self, sdfsfilename):
        self.membership_lock.acquire()
        index = hash(sdfsfilename) % len(self.membership_list)
        self.membership_lock.release()
        return index

    def getAllReplicas(self, index):
        indexes = set()
        self.membership_lock.acquire()
        l = len(self.membership_list)
        self.membership_lock.release()
        for i in range(0, 4):
            indexes.add((index + i) % l)
        self.membership_lock.acquire()
        res = [self.membership_list[i].split(':')[0] for i in indexes]
        self.membership_lock.release()
        return res

    def fileServerBackground(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.file_port))
            s.listen()
            while True:
                conn, addr = s.accept()
                t = threading.Thread(target=self.requestHandleThread, args=(conn, ))
                t.start()

    def requestHandleThread(self, conn: socket.socket):
        command = conn.recv(BUFFER_SIZE).decode()
        if command == 'put':
            t = threading.Thread(target=self.handle_put_request, args=(conn,))
            t.start()
        elif command == 'delete':
            t = threading.Thread(target=self.handle_delete_request, args=(conn,))
            t.start()
        elif command == 'get':
            t = threading.Thread(target=self.handle_get_request, args=(conn,))
            t.start()
        elif command == 'ls':
            t = threading.Thread(target=self.handle_ls_request, args=(conn,))
            t.start()
        elif command == 'repair':
            t = threading.Thread(target=self.handle_repair_request, args=(conn,))
            t.start()
        elif command == 'replicate':
            t = threading.Thread(target=self.handle_replicate_request, args=(conn,))
            t.start()
        elif command == 'multiget':
            t = threading.Thread(target=self.handle_multiple_get_request, args=(conn,))
            t.start()

    def handle_repair_request(self, conn: socket.socket):
        conn.send(b'1')
        encoded_command = conn.recv(BUFFER_SIZE)
        decoded_command = json.loads(encoded_command.decode())
        sdfsfileid, ips = decoded_command['sdfsfileid'], decoded_command['ips']
        if not self.file_table.check_file(sdfsfileid):
            conn.send(b'0')
        self.membership_lock.acquire()
        index = self.membership_list.index(self.id)
        for i in range(1, 4):
            ip = self.membership_list[(i+index)%len(self.membership_list)].split(':')[0]
            if ip not in ips:
                self.handle_replicate(sdfsfileid, ip)
                conn.send(b'1')
                break
        self.membership_lock.release()
        conn.send(b'0')
        return

    def handle_replicate(self, sdfsfileid, ip):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((ip, self.file_port))
            except socket.error as e:
                return
            s.send(b'replicate')
            s.recv(1)
            timestamp = self.file_table.check_file(sdfsfileid)
            localfilepath = sdfsfileid + '-' + str(timestamp)
            send_file(s, localfilepath, sdfsfileid, timestamp)
            s.recv(1)

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(json.dumps({'command_type': 'put_notice', 'command_content': [sdfsfileid, ip]}).encode(), (self.master_ip, self.master_port))

    def handle_replicate_request(self, conn):
        conn.send(b'1')
        data, sdfsfileid, timestamp = receive_file(conn)
        self.file_table.insert_file(data, sdfsfileid, timestamp)
        conn.send(b'1')

    def handle_put(self, localfilepath, sdfsfileid, ip, timestamp):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print('send to ', ip)
            try:
                s.connect((ip, self.file_port))
            except socket.error as e:
                return
            s.send(b'put')
            s.recv(1) # for ack
            send_file(s, localfilepath, sdfsfileid, timestamp)
            s.recv(1) # for ack
            command_id = sdfsfileid + '-' + str(timestamp)

            self.put_lock.acquire()
            self.put_ack_cache.setdefault(command_id, 0)
            self.put_ack_cache[command_id] += 1
            self.put_lock.release()

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(json.dumps({'command_type': 'put_notice', 'command_content': [sdfsfileid, ip]}).encode(), (self.master_ip, self.master_port))


    def handle_put_request(self, conn: socket.socket):
        conn.send(b'1')
        data, sdfsfileid, timestamp = receive_file(conn)
        self.file_table.insert_file(data, sdfsfileid, timestamp)
        conn.send(b'1')

    def handle_get(self, sdfsfileid, ip):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((ip, self.file_port))
            except socket.error as e:
                return
            s.send(b'get')
            s.recv(1)  # for ack
            s.send(sdfsfileid.encode())
            data, sdfsfileid, timestamp = receive_file(s)
            self.file_lock.acquire()
            self.file_cache.setdefault(sdfsfileid, [None, 0])
            if timestamp > self.file_cache[sdfsfileid][1]:
                self.file_cache[sdfsfileid] = [data, timestamp]
            self.file_lock.release()
            self.get_lock.acquire()
            self.get_ack_cache.setdefault(sdfsfileid, 0)
            self.get_ack_cache[sdfsfileid] += 1
            self.get_lock.release()

    def handle_get_request(self, conn: socket.socket):
        conn.send(b'1')
        sdfsfileid = conn.recv(BUFFER_SIZE).decode()
        timestamp = self.file_table.check_file(sdfsfileid)
        sdfsfilename = sdfsfileid + '-' + str(timestamp)
        send_file(conn, sdfsfilename, sdfsfileid, timestamp)

        return

    def handle_delete(self, sdfsfileid, ip):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((ip, self.file_port))
            except socket.error as e:
                return
            s.send(b'delete')
            s.recv(1)  # for ack
            s.send(sdfsfileid.encode())

    def handle_delete_request(self, conn: socket.socket):
        conn.send(b'1')
        sdfsfileid = conn.recv(BUFFER_SIZE).decode()
        self.file_table.delete_file(sdfsfileid)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(json.dumps({'command_type': 'delete_notice', 'command_content': [sdfsfileid, socket.gethostbyname(socket.gethostname())]}).encode(),
                     (self.master_ip, self.master_port))
        return

    def handle_ls(self, sdfsfileid, ip):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((ip, self.file_port))
            except socket.error as e:
                return
            s.send(b'ls')
            s.recv(1)  # for ack
            s.send(sdfsfileid.encode())
            res = s.recv(1).decode()
            if res == '1':
                self.ls_lock.acquire()
                self.ls_cache.setdefault(sdfsfileid, [])
                self.ls_cache[sdfsfileid].append(id)
                self.ls_lock.release()

    def handle_ls_request(self, conn: socket.socket):
        conn.send(b'1')
        sdfsfileid = conn.recv(BUFFER_SIZE).decode()
        exist = self.file_table.check_file(sdfsfileid)
        if exist:
            conn.send(b'1')
        else:
            conn.send(b'0')

    def handle_multiple_get(self, sdfsfileid, ip, n):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((ip, self.file_port))
            except socket.error as e:
                return
            s.send(b'multiget')
            s.recv(1) # for ack
            s.send(json.dumps({'sdfsfileid': sdfsfileid, 'n': n}).encode())
            obj = s.recv(4)
            header_size = struct.unpack('i', obj)[0]
            header_bytes = s.recv(header_size)
            header_json = header_bytes.decode()
            header_dic = json.loads(header_json)
            total_size, latest_t = header_dic['file_size'], header_dic['latest_t']
            data = b''
            recv_size = 0
            while recv_size < total_size:
                frag = s.recv(BUFFER_SIZE)
                data += frag
                recv_size += len(frag)
            k = sdfsfileid + '-' + str(n)
            self.file_lock.acquire()
            self.file_cache.setdefault(k, [None, 0])
            if self.file_cache[k][1] < latest_t:
                self.file_cache[k][0] = data
            self.file_lock.release()

            self.get_lock.acquire()
            self.get_ack_cache.setdefault(k, 0)
            self.get_ack_cache[k] += 1
            self.get_lock.release()

    def handle_multiple_get_request(self, conn: socket.socket):
        conn.send(b'1')
        decoded = json.loads(conn.recv(BUFFER_SIZE).decode())
        sdfsfileid, n = decoded['sdfsfileid'], decoded['n']
        data = self.file_table.get_n_versions(sdfsfileid, n)
        delimiter = b'\n\n-----------------------\n\n'
        data = delimiter.join(data)
        header_dic = {
            'file_size': len(data),
            'latest_t': self.file_table.check_file(sdfsfileid)
        }
        header_json = json.dumps(header_dic)
        header_bytes = header_json.encode()
        conn.send(struct.pack('i', len(header_bytes)))
        conn.send(header_bytes)
        conn.send(data)

    def run(self):
        self.join()
        t1 = threading.Thread(target=self.fileServerBackground)
        t1.start()


class MLInfer_Model:

    def __init__(self, file_server):
        self.file_server = file_server
        self.host = self.file_server.host
        self.append_port = APPEND_PORT
        self.infer_port = INFER_PORT
        self.model_init_port = MODEL_INIT_PORT
        self.batch_ack_port = BATCH_ACK_PORT
        self.send_model_init_port = MODEL_INIT_PORT
        self.job1_model = None
        self.job2_model = None
        self.job1_run = False
        self.job2_run = False
        self.job1_model_type = None
        self.job2_model_type = None
        self.client_total_queries = {}
        self.job1_outputfile = "job1_results.txt"
        self.job2_outputfile = "job2_results.txt"
        self.append_lock = threading.Lock()
        self.transform = transforms.Compose([            #[1]
        transforms.Resize(256),                    #[2]
        transforms.CenterCrop(224),                #[3]
        transforms.ToTensor(),                     #[4]
        transforms.Normalize(                      #[5]
        mean=[0.485, 0.456, 0.406],                #[6]
        std=[0.229, 0.224, 0.225]                  #[7]
        )])
        self.transform_init = False
        output_descr_hand = open('imagenet_class_index.json')
        self.output_descr = json.load(output_descr_hand)
        output_descr_hand.close()
        self.client_statistics = {}
        #self.set_batchsize_port = SET_BATCH_SIZE_PORT
        self.get_query_info_port = GET_QUERY_INFO_PORT
        self.get_vm_job_dict_port = GET_VM_JOB_DICT_PORT
        self.query_last_10secs_rate = {'job1': 0.49, 'job2': 0.07}
        self.job1_VMs_client = []
        self.job2_VMs_client = []
        self.job1_VMs_client_lock = threading.Lock()
        self.job2_VMs_client_lock = threading.Lock()

        trecieve_get_query_info = threading.Thread(target=self.recieve_get_query_info)
        trecieve_get_query_info.start()

        trecieve_vm_job_dict = threading.Thread(target=self.recieve_vm_job_dict)
        trecieve_vm_job_dict.start()

        trecieve_statistics = threading.Thread(target=self.receive_statistics)
        trecieve_statistics.start()

    def receive_statistics(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.file_server.host, STATISTICS_PORT))
            s.listen(3)
            while True:
                conn, addr = s.accept()
                obj = conn.recv(4)
                output_header_size = struct.unpack('i', obj)[0]
                output_bytes = conn.recv(output_header_size)
                output_json = output_bytes.decode()
                output_dic = json.loads(output_json)
                self.client_statistics = output_dic
                print(self.client_statistics)
        

    def recieve_vm_job_dict(self):
        # self.batch_size_set_lock.acquire()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.file_server.host, self.get_vm_job_dict_port))
            s.listen(3)
            while True:
                conn, addr = s.accept()
                obj = conn.recv(4)
                output_header_size = struct.unpack('i', obj)[0]
                output_bytes = conn.recv(output_header_size)
                output_json = output_bytes.decode()
                output_dic = json.loads(output_json)
                self.job1_VMs_client = output_dic['job1']
                self.job2_VMs_client = output_dic['job2']
                

    def recieve_get_query_info(self):
        # self.batch_size_set_lock.acquire()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.file_server.host, self.get_query_info_port))
            s.listen(3)
            while True:
                conn, addr = s.accept()
                obj = conn.recv(4)
                output_header_size = struct.unpack('i', obj)[0]
                output_bytes = conn.recv(output_header_size)
                output_json = output_bytes.decode()
                output_dic = json.loads(output_json)
                self.query_last_10secs_rate['job1'] = output_dic['job1']
                self.query_last_10secs_rate['job2'] = output_dic['job2']
        # self.batch_size_set_lock.release()

                    
    def run(self):
        treceive_init_model = threading.Thread(target=self.receive_start_model)
        treceive_init_model.start()
    
    def model_init(self, model_type, job_id):
        #print("Received model init")
        if job_id == 1:
            self.job1_run = True

            if model_type == "alexnet":
                self.job1_model_type = model_type

                self.job1_model = models.alexnet(pretrained = True)
            else:
                self.job1_model_type = model_type

                self.job1_model = models.resnet101(pretrained = True)
            
            trun_receive_model_output = threading.Thread(target=self.receive_model_output)
            trun_receive_model_output.start()

            treceive_image_list = threading.Thread(target=self.receive_image_list)
            treceive_image_list.start()
        
        else:
            self.job2_run = True

            if model_type == "alexnet":
                self.job2_model_type = model_type

                self.job2_model = models.alexnet(pretrained = True)
            else:
                self.job2_model_type = model_type

                self.job2_model = models.resnet101(pretrained = True)


    
    def receive_start_model(self):    # // To do start this thread --> Done
        #print("Receive Model Start")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.model_init_port)) #what is self.host
            s.listen(10)

            while True:
                conn, address = s.accept()
                obj = conn.recv(4)
                output_header_size = struct.unpack('i', obj)[0]
                output_bytes = conn.recv(output_header_size)
                output_json = output_bytes.decode()
                output_dic = json.loads(output_json)
                model_type = output_dic['mode_type']
                job_id = output_dic['job_id']

                # print(job_id)

                self.model_init(model_type, job_id)

    def send_model_output(self, vm_ip, output_str, output_filename):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((vm_ip, self.append_port))
            output_dict = {'filename': output_filename,
                               'model_output': output_str}
            output_json = json.dumps(output_dict)
            output_bytes = output_json.encode()
            s.send(struct.pack('i', len(output_bytes)))
            s.send(output_bytes)


    def append_to_output_file(self, client_socket, addr):
        obj = client_socket.recv(4)
        output_header_size = struct.unpack('i', obj)[0]
        output_bytes = client_socket.recv(output_header_size)
        output_json = output_bytes.decode()
        output_dic = json.loads(output_json)
        append_filename = output_dic['filename']
        append_output = output_dic['model_output']
        self.append_lock.acquire()

        file1 = open(append_filename, "a")
        file1.write(append_output)
        file1.close()
        self.append_lock.release()
        client_socket.close()

    def receive_model_output(self):
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as receive_server:
            receive_server.bind((self.host, self.append_port))
            receive_server.listen(10)

            while True:
                conn, address = receive_server.accept()

                toutput_append = threading.Thread(target=self.append_to_output_file, args = (conn, address,))
                toutput_append.start()

                #thread.start_new_thread(append_to_output_file, (conn, address, output_filename, output_str))

    def receive_image_list(self): # // To do start this thread once the model init starts
        print('Received image list:')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.infer_port))
            s.listen(3)

            while True:
                conn, address = s.accept()
                obj = conn.recv(4)

                output_header_size = struct.unpack('i', obj)[0]
                output_bytes = conn.recv(output_header_size)
                output_json = output_bytes.decode()
                output_dic = json.loads(output_json)
                job_id = output_dic['job_id']
                image_list = output_dic['image_list']

                if job_id == 1:
                    tinfer_job = threading.Thread(target=self.infer_job1, args = (image_list,))
                    tinfer_job.start()
                else:
                    tinfer_job = threading.Thread(target=self.infer_job2, args = (image_list,))
                    tinfer_job.start()


    def infer_job1(self, images_list):
        #print("inferring job 1")
        output_for_batch = ""
        for sdfsfileid in images_list:
            # print(images_list)
            localfilepath = DATASET_PATH + '/' + sdfsfileid

            self.job1_model.eval()

           
            filename = localfilepath
            img = Image.open(filename).convert('RGB')
            #print(len(img.shape))
            img_t = self.transform(img)
            # print(img_t.shape)
            batch_t = torch.unsqueeze(img_t, 0)
            out = self.job1_model(batch_t)
            ind = torch.argmax(out)

           
            output_for_batch += sdfsfileid + ", " + self.output_descr[str(int(ind))][1] + "\n"

        output_ips = [MASTER_HOST]

        for output_ip in output_ips:
            self.send_model_output(output_ip, output_for_batch, self.job1_outputfile)
            #os.system("echo \""+data[str(int(ind))][1] + "\" >"+ filename+"resnet_output")
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.file_server.master_ip, self.batch_ack_port))
                s.send(b'1')   


    def infer_job2(self, images_list):

        #print("inferring job 2")
        output_for_batch = ""
        for sdfsfileid in images_list:
            # print(images_list)
            localfilepath = DATASET_PATH + '/' + sdfsfileid

            self.job2_model.eval()

           
            filename = localfilepath
            img = Image.open(filename).convert('RGB')
            #print(len(img.shape))
            img_t = self.transform(img)
            # print(img_t.shape)
            batch_t = torch.unsqueeze(img_t, 0)
            out = self.job2_model(batch_t)
            ind = torch.argmax(out)

           
            output_for_batch += sdfsfileid + ", " + self.output_descr[str(int(ind))][1] + "\n"

        output_ips = [MASTER_HOST]

        for output_ip in output_ips:
            self.send_model_output(output_ip, output_for_batch, self.job2_outputfile)
            #os.system("echo \""+data[str(int(ind))][1] + "\" >"+ filename+"resnet_output")
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.file_server.master_ip, self.batch_ack_port))
                s.send(b'2')  


class MLCoordinator_Server:

    def __init__(self, file_server):
        self.file_server = file_server
        # self.host = self.file_server.host
        self.is_load_bal_run = False
        # self.vm_list = self.file_server.membership_list
        
        self.job1_scheduled = False
        self.job2_scheduled = False
        self.job1_VMs = []
        self.job2_VMs = []
        self.job1_VM_lock = threading.Lock()
        self.job2_VM_lock = threading.Lock()
        self.cache_image_run = {}
        self.cache_image_run_lock = threading.Lock()
        self.total_no_queries_processed = {}
        self.total_query_proc_lock = threading.Lock()
        self.no_of_queries_processed = {}
        # self.no_of_queries_processed = defaultdict(dict)
        # for vm_ip in ALL_VMS:
        #     self.no_of_queries_processed['job1'][vm_ip] = 0
        #     self.no_of_queries_processed['job2'][vm_ip] = 0
        self.no_of_query_process_lock = threading.Lock()
        
        self.query_rate = {}
        self.query_last_10secs_rate = {}
        self.query_rate_forever = {}
        self.statistics = {}
        # self.batch_size_set_lock = threading.Lock()
        self.batch_size = 2
        self.batch_ack_port = BATCH_ACK_PORT
        self.send_model_init_port = MODEL_INIT_PORT
        self.infer_port = INFER_PORT
        self.total_no_images = 2500
        # self.set_batchsize_port = SET_BATCH_SIZE_PORT
        self.get_query_info_port = GET_QUERY_INFO_PORT
        self.get_vm_job_dict_port = GET_VM_JOB_DICT_PORT

        tsend_get_query_info = threading.Thread(target=self.send_get_query_info)
        tsend_get_query_info.start()

        tsend_statistics = threading.Thread(target=self.send_statistics)
        tsend_statistics.start()

        #treceive_set_batch_size = threading.Thread(target=self.receive_set_batch_size)
        #treceive_set_batch_size.start()

        tsend_get_vm_job_dict = threading.Thread(target=self.send_get_vm_job_dict)
        tsend_get_vm_job_dict.start()

        tsend_total_queries_when_asked = threading.Thread(target=self.send_total_queries_when_asked)
        tsend_total_queries_when_asked.start()

        self.job1_outputfile = "job1_results.txt"
        self.job2_outputfile = "job2_results.txt"

    
    def update_statistics(self):
        self.query_rate_forever.setdefault('job1', [])
        self.query_rate_forever.setdefault('job2', [])
        count = 0

        self.statistics.setdefault('average_job1', 0.0)
        self.statistics.setdefault('sd_job1', 0.0)
        self.statistics.setdefault('median_job1', 0.0)
        self.statistics.setdefault('90th_job1', 0.0)
        self.statistics.setdefault('95th_job1', 0.0)
        self.statistics.setdefault('99th_job1', 0.0)
        self.statistics.setdefault('median_job1', 0.0)

        self.statistics.setdefault('average_job2', 0.0)
        self.statistics.setdefault('sd_job2', 0.0)
        self.statistics.setdefault('median_job2', 0.0)
        self.statistics.setdefault('90th_job2', 0.0)
        self.statistics.setdefault('95th_job2', 0.0)
        self.statistics.setdefault('99th_job2', 0.0)
        self.statistics.setdefault('median_job2', 0.0)
        
        while True:
            time.sleep(20)
            count += 1
            self.query_rate_forever['job1'].append(self.query_rate['job1'])
            self.query_rate_forever['job2'].append(self.query_rate['job2'])
            print(self.query_rate_forever)

            if len(self.query_rate_forever['job1']) == 0 or len(self.query_rate_forever['job2']) == 0:
                continue

            self.statistics['average_job1'] = np.mean(self.query_rate_forever['job1'])
            print(self.statistics['average_job1'])
            self.statistics['sd_job1'] = np.std(self.query_rate_forever['job1'])
            self.statistics['90th_job1']  = np.percentile(self.query_rate_forever['job1'], 90)
            self.statistics['95th_job1'] = np.percentile(self.query_rate_forever['job1'], 95)
            self.statistics['99th_job1'] = np.percentile(self.query_rate_forever['job1'], 99)
            self.statistics['median_job1'] = np.percentile(self.query_rate_forever['job1'], 50)

            self.statistics['average_job2'] = np.mean(self.query_rate_forever['job2'])
            self.statistics['sd_job2'] = np.std(self.query_rate_forever['job2'])
            self.statistics['90th_job2']  = np.percentile(self.query_rate_forever['job2'], 90)
            self.statistics['95th_job2'] = np.percentile(self.query_rate_forever['job2'], 95)
            self.statistics['95th_job2'] = np.percentile(self.query_rate_forever['job2'], 99)
            self.statistics['95th_job2'] = np.percentile(self.query_rate_forever['job2'], 50)

    def send_total_queries_when_asked(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.file_server.host, TOTAL_QUERIES_PORT))
            s.listen(3)

            while True:
                conn, address = s.accept()
                obj = conn.recv(1)

                output_dict = self.total_no_queries_processed
                output_json = json.dumps(output_dict)
                output_bytes = output_json.encode()
                conn.send(struct.pack('i', len(output_bytes)))
                conn.send(output_bytes)


    def set_batch_size(self, input_batchsize):
        # self.batch_size_set_lock.acquire()
        self.batch_size = input_batchsize

    def send_get_vm_job_dict(self):
        # self.batch_size_set_lock.acquire()
        time.sleep(4)
        while True:
            for vm in self.file_server.membership_list:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        vm_ip = vm.split(':')[0] #To get IP address take first 10 indices
                        #print(vm_ip)
                        s.connect((vm_ip, self.get_vm_job_dict_port))
                        self.job1_VM_lock.acquire()
                        self.job2_VM_lock.acquire()
                        job1_VMs_list = self.job1_VMs
                        job2_VMs_list = self.job2_VMs
                        self.job1_VM_lock.release()
                        self.job2_VM_lock.release()
                        job_dict = {'job1': job1_VMs_list, 'job2': job2_VMs_list}
                        
                        output_json = json.dumps(job_dict)
                        output_bytes = output_json.encode()
                        s.send(struct.pack('i', len(output_bytes)))
                        s.send(output_bytes)
                        
            time.sleep(4)


    def send_get_query_info(self):
        # self.batch_size_set_lock.acquire()
        time.sleep(4)
        while True:
            for vm in self.file_server.membership_list:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    vm_ip = vm.split(':')[0] #To get IP address take first 10 indices
                    s.connect((vm_ip, self.get_query_info_port))
                    query_dict = self.query_rate
                    query_dict.setdefault('job1', 0)
                    query_dict.setdefault('job2', 0)
                    output_json = json.dumps(query_dict)
                    output_bytes = output_json.encode()
                    s.send(struct.pack('i', len(output_bytes)))
                    s.send(output_bytes)
            time.sleep(4)
        # self.batch_size_set_lock.release()

    def send_statistics(self):
        # self.batch_size_set_lock.acquire()
        time.sleep(30)
        while True:
            for vm in self.file_server.membership_list:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    vm_ip = vm.split(':')[0] #To get IP address take first 10 indices
                    s.connect((vm_ip, STATISTICS_PORT))
                    
                    output_json = json.dumps(self.statistics)
                    # print(self.statistics)
                    output_bytes = output_json.encode()
                    s.send(struct.pack('i', len(output_bytes)))
                    s.send(output_bytes)
            time.sleep(10)
    
    def start_model(self,vm_list, model_type, job_id):
        for vm in vm_list:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                vm_ip = vm.split(':')[0]
                # vm = vm[:10] #To get IP address take first 10 indices
                #print((vm_ip, self.send_model_init_port))
                s.connect((vm_ip, self.send_model_init_port))
                output_dict = {'mode_type': model_type,
                               'job_id': job_id}
                output_json = json.dumps(output_dict)
                output_bytes = output_json.encode()
                s.send(struct.pack('i', len(output_bytes)))
                s.send(output_bytes)


    def schedule_job(self, model_type):
        print('Schedule job:')
        job_id = None
        if not self.is_load_bal_run:
            tload_schedule = threading.Thread(target=self.load_balancer)
            tload_schedule.start()
            self.is_load_bal_run = True
        
        if not self.job1_scheduled:
            self.job1_scheduled = True
            for member_ip in self.file_server.membership_list:
                member_ip_extract = member_ip.split(':')[0]
                self.job1_VMs.append(member_ip_extract)
            print(self.job1_VMs)
            job_id = 1
            self.total_query_proc_lock.acquire()
            self.total_no_queries_processed.setdefault('job1', 0)
            self.total_query_proc_lock.release()

            treceive_batch_ack = threading.Thread(target=self.receive_batch_ack)
            treceive_batch_ack.start()

        else:
            self.job2_scheduled = True
            for member_ip in self.file_server.membership_list:
                member_ip_extract = member_ip.split(':')[0]
                self.job2_VMs.append(member_ip_extract)
            
            job_id = 2
            self.total_query_proc_lock.acquire()
            self.total_no_queries_processed.setdefault('job2', 0)
            self.total_query_proc_lock.release()

        
        self.start_model(self.file_server.membership_list, model_type, job_id)

        
        trun_query = threading.Thread(target=self.run_query, args = (job_id, model_type,))
        trun_query.start()  

    
    def send_image_list(self, vm_ip, job_id, image_list):
        # print("Send image list: ")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((vm_ip, self.infer_port))
                output_dict = {'job_id': job_id,
                               'image_list': image_list}
                output_json = json.dumps(output_dict)
                output_bytes = output_json.encode()
                s.send(struct.pack('i', len(output_bytes)))
                s.send(output_bytes)

    def run_query(self, job_id, model_type):
        print('Run query:')
        image_count = 1
        query_vms = []

        run_count = 1

        time.sleep(10)

        while image_count <= 2500:     # MAKE IT TO 2500
            batch_size_cur = self.batch_size
            image_string = 'ILSVRC2012_test_'
            if job_id == 1:
                self.job1_VM_lock.acquire()
                query_vms = self.job1_VMs
                self.job1_VM_lock.release()
            else:
                self.job2_VM_lock.acquire()
                query_vms = self.job2_VMs
                self.job2_VM_lock.release()
            # print(query_vms)
            for vm in query_vms:
                # print('Inside run_query: ')
                
                if job_id == 1:
                    self.cache_image_run_lock.acquire()
                    self.cache_image_run.setdefault('job1', {}).setdefault(str(vm), [])
                    job1_vm_image_list = self.cache_image_run['job1'][str(vm)]
                    #print(job1_vm_image_list)
                    self.cache_image_run_lock.release()
                    if len(job1_vm_image_list) != 0 and run_count == 0:
                        #print('Not intended here')
                        continue
                else:
                    self.cache_image_run_lock.acquire()
                    self.cache_image_run.setdefault('job2', {}).setdefault(str(vm), [])
                    job2_vm_image_list = self.cache_image_run['job2'][str(vm)]
                    self.cache_image_run_lock.release()
                    if len(job2_vm_image_list) != 0 and run_count == 0:
                        continue
                
                image_list = []
                for i in range(batch_size_cur):
                    append_count = str(image_count+i)
                    str_count = append_count.zfill(8)
                    image = image_string + str_count + '.JPEG'
                    image_list.append(image)
                #print(image_list)
                self.cache_image_run_lock.acquire()
                if job_id == 1:
                    self.cache_image_run['job1'][str(vm)] = image_list
                else:
                    self.cache_image_run['job2'][str(vm)] = image_list
                self.cache_image_run_lock.release()
                image_count += batch_size_cur
                self.send_image_list(vm, job_id, image_list)
            run_count = 0
    
    def process_batch_ack(self, conn, addr):
        #print("address:", addr)
        job_id = conn.recv(1).decode()   #ack
        if job_id == '1':                # // To do check this -- > Yet to be done
            self.no_of_query_process_lock.acquire()
            self.no_of_queries_processed.setdefault('job1', {})
            self.no_of_queries_processed['job1'].setdefault(str(addr[0]), 0)
            self.no_of_queries_processed['job1'][str(addr[0])] += self.batch_size
            self.no_of_query_process_lock.release()
            self.total_query_proc_lock.acquire()
            self.total_no_queries_processed['job1'] += self.batch_size
            self.total_query_proc_lock.release()
            self.cache_image_run_lock.acquire()
            #print(self.cache_image_run['job1'])
            self.cache_image_run['job1'][str(addr[0])] = []
            self.cache_image_run_lock.release()

        else:
            self.no_of_query_process_lock.acquire()
            self.no_of_queries_processed.setdefault('job2', {}).setdefault(str(addr[0]), 0)
            self.no_of_queries_processed['job2'].setdefault(str(addr[0]), 0)
            self.no_of_queries_processed['job2'][str(addr[0])] += self.batch_size
            self.no_of_query_process_lock.release()
            self.total_query_proc_lock.acquire()
            self.total_no_queries_processed['job2'] += self.batch_size
            self.total_query_proc_lock.release()
            self.cache_image_run_lock.acquire()
            self.cache_image_run['job2'][str(addr[0])] = []
            self.cache_image_run_lock.release()
        
    def receive_batch_ack(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.file_server.host, self.batch_ack_port))
            s.listen(10)

            while True:
                conn, address = s.accept()

                toutput_append = threading.Thread(target=self.process_batch_ack, args = (conn, address,))
                toutput_append.start()

    def update_VM_list(self):
        time.sleep(3)
        #print("Came here")

        while True:
            membership_list_extract = []
            for member_ip in self.file_server.membership_list:
                member_ip_extract = member_ip.split(':')[0]
                membership_list_extract.append(member_ip_extract)

            for vm_ip in self.job2_VMs:
                if vm_ip not in membership_list_extract:
                    print("Came here")
                    self.job2_VMs.remove(vm_ip)

            time.sleep(3)


    def load_balancer(self):
        print('Load Balancer:')
        self.query_rate['job1'] = 0
        self.query_rate['job2'] = 0
        self.no_of_query_process_lock.acquire()
        self.no_of_queries_processed.setdefault('job1', {})
        self.no_of_queries_processed.setdefault('job2', {})
        # print('load_balancer:', self.no_of_queries_processed['job1'])
        self.no_of_query_process_lock.release()
        time.sleep(2)

        tupdate_VM_thread = threading.Thread(target=self.update_VM_list)
        tupdate_VM_thread.start()

        tupdate_statistics = threading.Thread(target=self.update_statistics)
        tupdate_statistics.start()

        while True:     
            self.no_of_query_process_lock.acquire()
            job1_queries_processed = 0
            job2_queries_processed = 0

            for vm in self.no_of_queries_processed['job1'].keys():
                job1_queries_processed += self.no_of_queries_processed['job1'][vm]
            for vm in self.no_of_queries_processed['job2'].keys():
                job2_queries_processed += self.no_of_queries_processed['job2'][vm]
            self.no_of_query_process_lock.release()
            time.sleep(2)
            job1_queries_processed_after_2_sec = 0
            job2_queries_processed_after_2_sec = 0
            self.no_of_query_process_lock.acquire()
            for vm in self.no_of_queries_processed['job1'].keys():
                job1_queries_processed_after_2_sec += self.no_of_queries_processed['job1'][vm]
            for vm in self.no_of_queries_processed['job2'].keys():
                job2_queries_processed_after_2_sec += self.no_of_queries_processed['job2'][vm]
            self.no_of_query_process_lock.release()
            self.query_rate['job1'] = (job1_queries_processed_after_2_sec - job1_queries_processed)/2.0
            self.query_rate['job2'] = (job2_queries_processed_after_2_sec - job2_queries_processed)/2.0

            if 1.2*self.query_rate['job1'] > self.query_rate['job2']:
                if len(self.job1_VMs) > 1:
                    self.job1_VMs.pop()
            elif 1.2*self.query_rate['job1'] < self.query_rate['job2']:
                count = 0
                if len(self.job2_VMs) > len(self.job1_VMs):
                    for vm_id in self.job2_VMs:
                        if vm_id not in self.job1_VMs and count < 1:
                            self.job1_VMs.append(vm_id)
                            count += 1

    def last_10_secs_query_rate(self):
        while True:
            time.sleep(10)
            self.no_of_query_process_lock.acquire()
            job1_last_10secs_queries_processed = 0
            job2_last_10secs_queries_processed = 0
            for vm in self.no_of_queries_processed['job1'].keys():
                job1_last_10secs_queries_processed += self.no_of_queries_processed['job1'][vm]
            for vm in self.no_of_queries_processed['job2'].keys():
                job2_last_10secs_queries_processed += self.no_of_queries_processed['job2'][vm]
            self.no_of_query_process_lock.release()
            self.query_last_10secs_rate['job1'] = job1_last_10secs_queries_processed/10.0
            self.query_last_10secs_rate['job2'] = job2_last_10secs_queries_processed/10.0


if __name__ == '__main__':
    # def __init__(self, ping_port: int, membership_port: int, ping_timeout: int, ping_interval: int, log_filepath: str,
    #              file_port: int):
    server = FServer(PING_PORT, MEMBERSHIP_PORT, PING_TIMEOUT, PING_INTERVAL, LOG_FILEPATH, FILE_PORT, MASTER_PORT, MASTER_HOST)
    file_thread = threading.Thread(target=server.run)
    file_thread.start()
    # server.run()
    ml_server = None
    if server.isIntroducer:
        ml_server = MLCoordinator_Server(server)
        #ml_server.run()
    
    ml_vm = MLInfer_Model(server)
    ml_vm.run()

    job_count = 1
    if server.isIntroducer:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((ml_server.file_server.host, SCHEDULE_JOB_PORT))
            s.listen(3)

            while job_count <= 2:
                while True:
                    conn, address = s.accept()
                    job_id = conn.recv(1).decode()

                    if job_count == 1:
                        ml_server.schedule_job('alexnet')
                    else:
                        ml_server.schedule_job('resnet')
                    job_count += 1
                    if job_count == 3:
                        break 

    while True:
            command = input('>')
            parsed_command = command.split()
            start_time = time.time()


            if parsed_command[0] == 'query_info':
                print("Job1", ml_vm.query_last_10secs_rate['job1'])
                print("Job2", ml_vm.query_last_10secs_rate['job2'])
            elif command == 'list_mem':
                print('isIntroducer: ', server.isIntroducer)
                server.membership_lock.acquire()
                print(f'there are {len(server.membership_list)} member in membership_list: ')
                for member in server.membership_list:
                    print('    ', member)
                server.membership_lock.release()
                server.command_lock.acquire()
                print(f'{len(server.commands)} commands have been executed')
                server.command_lock.release()
            elif parsed_command[0] == 'put_images':
                dataset_path = DATASET_PATH
                image_list = os.listdir(dataset_path)
                image_count = 0
                for fi in image_list:
                    fi_path = dataset_path + '/' + fi
                    sdfsfileid = fi
                    ips = server.get_ip(sdfsfileid)
                    if not ips:
                        index = server.filehash(sdfsfileid)
                        ips = server.getAllReplicas(index)
                    timestamp = time.time()
                    for ip in ips:
                        t = threading.Thread(target=server.handle_put, args = (fi_path, sdfsfileid, ip, timestamp))
                        t.start()
                    command_id = sdfsfileid + '-' + str(timestamp)
                    while True:
                        server.put_lock.acquire()
                        server.put_ack_cache.setdefault(command_id, 0)
                        cnt = server.put_ack_cache[command_id]
                        server.put_lock.release()
                        if cnt >= 3:
                            break
                    image_count += 1
                    if image_count >= 10:
                        break
                print('put images complete.')

            elif parsed_command[0] == 'processing_time':
                print('')
            
            elif parsed_command[0] == 'vm_job_dict':
                print("Job1 ", ml_server.job1_VMs)
                print("Job2 ", ml_server.job2_VMs)
                #create client server code for setting batch size

            elif parsed_command[0] == 'set_batchsize':
                if server.isIntroducer:
                    ml_server.batch_size = parsed_command[1]
                else:
                    ml_vm.send_set_batch_size(int(parsed_command[1]))
            elif parsed_command[0] == 'infer':
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((MASTER_HOST, SCHEDULE_JOB_PORT))
                    s.send(b'1')  
            elif parsed_command[0] == 'total_queries':
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((MASTER_HOST, TOTAL_QUERIES_PORT))
                    s.send(b'1')
                    time.sleep(2)
                    obj = s.recv(4)

                    output_header_size = struct.unpack('i', obj)[0]
                    output_bytes = s.recv(output_header_size)
                    output_json = output_bytes.decode()
                    output_dic = json.loads(output_json)
                    print(output_dic)

            elif parsed_command[0] == 'print_statistics':
                print(ml_server.statistics)
            elif parsed_command[0] == 'get':
                sdfsfileid, localfilepath = parsed_command[1], parsed_command[2]
                ips = server.get_ip(sdfsfileid)
                print(len(ips))
                for ip in ips:
                    t = threading.Thread(target=server.handle_get, args=(sdfsfileid, ip))
                    t.start()
                i = 0

                get_start_time = time.time()
                get_fail = False

                while True:
                    curr_time = time.time()

                    if curr_time - get_start_time > 20:
                        get_fail = True
                        break
                    
                    server.get_lock.acquire()
                    server.get_ack_cache.setdefault(sdfsfileid, 0)
                    cnt = server.get_ack_cache[sdfsfileid]
                    server.get_lock.release()
                    if cnt >= 3:
                        break

                if get_fail:
                    print('get failed.')
                else:
                    server.file_lock.acquire()
                    data = server.file_cache[sdfsfileid][0]
                    server.file_lock.release()
                    with open(localfilepath, 'wb') as f:
                        f.write(data)
                    print('get complete.')
            else:
                print('command not found!')
            end_time = time.time()
            print('time consumed: ', end_time - start_time)



