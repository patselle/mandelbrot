from ctypes import *
import json
import time
import requests
import threading
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--url', type=str, default='http://127.0.0.1:5000', help='thread count')
parser.add_argument('--threads', type=int, default=1, help='thread count')

args = parser.parse_args()
lib =  CDLL('core/lib/libpyfrac.so')
threads = []

def thread_run(name):
    global args, lib

    while True:
        response = requests.get(f'{args.url}/work')

        if not response.status_code == 200:
            print("response did not indicate success")
            break

        work = json.loads(response._content.decode('utf-8'))

        if work['id'] == None:
            print('sleep')
            time.sleep(5)
            continue

        config =  work['config']

        length = config['width'] * config['row_height']
        frame = (c_uint * length)()
        lib.pyfrac_frame(pointer(frame), config['width'], config['height'], work['id'], config['row_height'], config['r_min'].encode('utf-8'), config['r_max'].encode('utf-8'), config['i_min'].encode('utf-8'), config['i_max'].encode('utf-8'), config['precision'], config['threshold'], config['limit'])

        l = [frame[i] for i in range(length)]
        requests.post(f'{args.url}/work', json={ 'id': work['id'], 'frame': l })

for x in range(0, args.threads):
    t = threading.Thread(target = thread_run, args = (x,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()