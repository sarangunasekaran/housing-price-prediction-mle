import os
import subprocess

dirpath = os.path.dirname(os.path.abspath(__file__))
sh_path = os.path.join(dirpath, "local_vm_setup.sh")
print(sh_path)

subprocess.call(['sh', "local_vm_setup.sh"], shell=True)
