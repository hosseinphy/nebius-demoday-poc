# Cluster:
`sinfo:`

```bash
PARTITION  AVAIL  TIMELIMIT  NODES  STATE NODELIST
main*         up   infinite      2   idle worker-[0-1]
background    up   infinite      2   idle worker-[0-1]
```

---

`scontrol show nodes:`

```bash
NodeName=worker-1 Arch=x86_64 CoresPerSocket=32 
   CPUAlloc=0 CPUEfctv=128 CPUTot=128 CPULoad=0.67
   AvailableFeatures=(null)
   ActiveFeatures=(null)
   Gres=gpu:nvidia_h100_80gb_hbm3:8(S:0-1)
   NodeAddr=10.5.170.103 NodeHostName=worker-1 Version=24.11.5
   OS=Linux 5.15.0-164-generic #174-Ubuntu SMP Fri Nov 14 20:25:16 UTC 2025 
   RealMemory=1570840 AllocMem=0 FreeMem=1572718 Sockets=2 Boards=1
   State=IDLE+DYNAMIC_NORM ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=main,background 
   BootTime=2026-01-14 13:48:23.UTC SlurmdStartTime=2026-01-14 14:04:46.UTC
   LastBusyTime=2026-01-14 13:58:13.UTC ResumeAfterTime=None
   CfgTRES=cpu=128,mem=1570840M,billing=128,gres/gpu=8
   AllocTRES=
   CurrentWatts=0 AveWatts=0
   
   InstanceId=computeinstance-e00rcsvhgse853pyd2 

NodeName=worker-0 Arch=x86_64 CoresPerSocket=32 
   CPUAlloc=0 CPUEfctv=128 CPUTot=128 CPULoad=1.04
   AvailableFeatures=(null)
   ActiveFeatures=(null)
   Gres=gpu:nvidia_h100_80gb_hbm3:8(S:0-1)
   NodeAddr=10.5.171.41 NodeHostName=worker-0 Version=24.11.5
   OS=Linux 5.15.0-164-generic #174-Ubuntu SMP Fri Nov 14 20:25:16 UTC 2025 
   RealMemory=1570840 AllocMem=0 FreeMem=1556404 Sockets=2 Boards=1
   State=IDLE+DYNAMIC_NORM ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=main,background 
   BootTime=2026-01-14 13:48:20.UTC SlurmdStartTime=2026-01-14 14:04:46.UTC
   LastBusyTime=2026-01-14 13:58:14.UTC ResumeAfterTime=None
   CfgTRES=cpu=128,mem=1570840M,billing=128,gres/gpu=8
   AllocTRES=
   CurrentWatts=0 AveWatts=0
   
   InstanceId=computeinstance-e00jsy05kxfa68pbmb 

```

`sbatch discover_env.sbatch`:


```bash
== Hostname ===
worker-1
=== GPU check ===
Wed Jan 14 14:42:06 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.148.08             Driver Version: 570.148.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:8D:00.0 Off |                    0 |
| N/A   28C    P0             70W /  700W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
=== Filesystems ===
Filesystem      Size  Used Avail Use% Mounted on
jail            5.0T   15G  5.0T   1% /
overlay         2.0T   30G  2.0T   2% /run
tmpfs            64M     0   64M   0% /dev
tmpfs            64G     0   64G   0% /dev/shm
/dev/vda1       2.0T   30G  2.0T   2% /etc/hosts
tmpfs            64M     0   64M   0% /dev/nvidia2
tmpfs           158G   97M  158G   1% /run/nvidia
overlay         2.0T   30G  2.0T   2% /run/nvidia/driver
shm              64M     0   64M   0% /run/nvidia/driver/dev/shm
overlay         2.0T   30G  2.0T   2% /run/nvidia/driver/run/mellanox/drivers/usr/src
tmpfs           1.6T   12K  1.6T   1% /run/nvidia/driver/run/secrets/kubernetes.io/serviceaccount
tmpfs           1.5T   12K  1.5T   1% /run/secrets/kubernetes.io/serviceaccount
tmpfs           1.5T     0  1.5T   0% /mnt/memory


=== Mount ===
<Too long to show>

=== Root dir ===
bin
boot
dev
etc
home
lib
lib32
lib64
libx32
media
mnt
opt
proc
root
run
sbin
srv
sys
sys-host
tmp
usr
var

```

From the info above I can infer the followings for GPU and  storage initially:
## GPU:

```
Worker node: worker-1
GPU: NVIDIA H100 80GB HBM3
Driver: 570.148.08
CUDA: 12.8
MIG: Disabled
```

## Storage:
```
Large filesystem appears as jail ~5TB mounted at /
2TB device /dev/vda1 visible (likely local/ephemeral)
```
Need to confirm which mounts correspond to “2TB network disk” and “2TB shared FS” using df -hT and findmnt




<!-- 
- Scheduler: Slurm (Soperator)
- Nodes: 2
- GPUs per node: 8
- GPU type: NVIDIA H100

# Storage:
- Shared filesystem: <path>
- Network disk: <path> -->
