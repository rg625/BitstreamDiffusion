#!/bin/bash
# Inter-node NCCL over Isambard-AI's Slingshot fabric. Sourced by launchers only
# when a job spans more than one node; single-node jobs never touch it, so
# every existing run is unaffected.
#
# Without the libfabric plugin NCCL falls back to TCP sockets across nodes:
# it WORKS, and is an order of magnitude slower, so a multi-node run would
# look healthy while losing most of its speedup. NCCL_NET below makes a missing
# plugin a hard error instead of a silent fallback.
#
# The plugin is put on the library path by hand rather than with
# `module load brics/aws-ofi-nccl`: that module also loads brics/nccl 2.26,
# whose libnccl on LD_LIBRARY_PATH would shadow the NCCL 2.27 that torch ships
# and was tested with. Only the net plugin and libfabric are taken from the
# system.
if [ "${COBIT_SITE:-}" = "isambard" ]; then
  _ofi=/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/aws-ofi-nccl-1.8.1-c47cd5ivrugm3jzlyqyis4igyflnydmo
  [ -f "$_ofi/lib/libnccl-net.so" ] || { echo "[multinode] FATAL: no aws-ofi-nccl plugin at $_ofi" >&2; exit 1; }
  export LD_LIBRARY_PATH="$_ofi/lib:/opt/cray/libfabric/1.22.0/lib64:${LD_LIBRARY_PATH:-}"
  export NCCL_NET="AWS Libfabric"
  export FI_PROVIDER=cxi
  export NCCL_SOCKET_IFNAME=hsn          # bootstrap traffic on the high-speed NICs
  export GLOO_SOCKET_IFNAME=hsn0
  export NCCL_CROSS_NIC=1
  export NCCL_NET_GDR_LEVEL=PHB
  export FI_CXI_DISABLE_HOST_REGISTER=1
  export FI_MR_CACHE_MONITOR=userfaultfd
  export FI_CXI_DEFAULT_CQ_SIZE=131072
fi

# torchrun rendezvous: the first node of the allocation hosts the store.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"
echo "[multinode] nodes=$SLURM_JOB_NUM_NODES master=$MASTER_ADDR:$MASTER_PORT net=${NCCL_NET:-default}"
