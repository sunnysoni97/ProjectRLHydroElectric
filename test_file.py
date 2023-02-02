import shutil
import os

if __name__ == "__main__":
    dst_file = os.path.join(os.path.dirname(__file__),'requirements.txt')
    src_file = os.path.join(os.path.dirname(__file__),'rough_space/requirements.txt')
    shutil.copy2(src=src_file,dst=dst_file)
