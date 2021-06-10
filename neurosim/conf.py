import json
import sys
import getpass

fnjson = 'config.json'

for i in range(len(sys.argv)):
  if sys.argv[i].endswith('.json'):
    fnjson = sys.argv[i]
    print('reading ', fnjson)

def readconf(fnjson):
  with open(fnjson, 'r') as fp:
    dconf = json.load(fp)
  return dconf

dconf = readconf(fnjson) # read the configuration
