Copy the memory dump you want to scan into this folder (for example a
VirtualBox .elf from "VBoxManage debugvm ... dumpvmcore", a VMware .vmem
together with its .vmss/.vmsn file, or a raw .raw/.mem image), then press
Refresh in the HMS window.

The first scan of a dump creates a <dump name>_vol folder next to it with
Volatility's output, so later scans of the same dump are quick.
