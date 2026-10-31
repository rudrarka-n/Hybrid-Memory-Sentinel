Your own memory dumps for training the CNN (no public dataset exists).

  benign\      dumps of the VM with only normal programs running
  malicious\   dumps taken after running a malware sample in the VM;
               optionally one subfolder per family, e.g. malicious\Ransomware\

Put at least two dumps in each folder (more is better), all captured the same
way from the same VM. Then press "Train CNN" in the Train tab.

Never run malware outside an isolated VM. A dump is only data: copying it
here does not run anything.
