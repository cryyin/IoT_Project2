# This is a sample Python script.
import os
from os import path
import pathlib
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


    source = path.normpath(r'C:\Users\chunyuan\Downloads\spi')
    Cur_dir = pathlib.Path(os.path.join(os.getcwd()))
    data_dir = pathlib.Path(os.path.join(os.getcwd(), 'spi'))
    output_path = 'spice_output\\'
    videoList = os.listdir(data_dir)

    '''
    for Sname in data_dir:
        if not Sname.endswith("wav"):
            videoList.remove(Sname)'''

    for i in videoList:
        output = i[0:-4]
        cmd = "ffmpeg -i %s\\%s -ab 256k -ar 16000 -t 1 %s\\output_spice\\output_%s.wav" % (data_dir,i,Cur_dir, output)
        print(cmd)
        os.system(cmd)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
