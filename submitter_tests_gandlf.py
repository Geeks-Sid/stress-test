import subprocess

batch_sizes=[6,24,96,384,1536]
thread_sizes=[1,2,4,8,16,32,64]

for b in batch_sizes:
  for t in thread_sizes:
    command = 'python gandlf-test-v1.py -i ~/projects_wsl/stress-test/data.csv -b ' + str(b) + ' -t ' + str(t)
    # print(command)
    subprocess.Popen(command, shell=True).wait()
