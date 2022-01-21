#!/usr/bin/env python
# coding: utf-8

import sys
import requests
import json

def main():
    text = sys.argv[1]

    TOKEN = 'xoxb-2337202390117-2340473666611-29HxNDC9qKFAe7bpkP2cjimF'
    CHANNEL = 'expe'

    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": "Bearer "+TOKEN}
    data  = {
    'channel': CHANNEL,
    'text': text
    }
    r = requests.post(url, headers=headers, data=data)
    #print("return ", r.json())

if __name__ == '__main__':
    main()