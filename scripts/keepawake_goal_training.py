"""Temporarily prevent system sleep while one authorized training process runs.

No display wake request, persistent power-plan change, or child training process.
The Windows request is released automatically when this helper exits.
"""
import argparse
import ctypes
import sys
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);args=parser.parse_args()
    if sys.platform!='win32':return
    k=ctypes.windll.kernel32
    k.OpenProcess.restype=ctypes.c_void_p
    k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    k.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=k.OpenProcess(0x00100000,False,args.pid)
    if not handle:raise OSError('Training process is not available')
    try:
        if not k.SetThreadExecutionState(0x80000001):raise OSError('Could not request system availability')
        while k.WaitForSingleObject(handle,30000)==0x00000102:
            pass
    finally:
        k.SetThreadExecutionState(0x80000000);k.CloseHandle(handle)

if __name__=='__main__':main()
