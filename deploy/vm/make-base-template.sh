#!/bin/bash
# deploy/vm/make-base-template.sh — EXPERIMENTAL (not yet exercised end-to-end;
# (the reference base template was built by hand) — build a frozen
# default-state base template for oneshot.sh from an Ubuntu cloud image.
#
# Prerequisites: libvirt, qemu-utils, and cloud-localds (cloud-image-utils) or
# genisoimage/xorriso/mkisofs for the cloud-init seed. Internet access to
# download the cloud image unless --cloud-image points at an existing one.
#
# Usage: make-base-template.sh [flags] [--cloud-image FILE] [--arch amd64|arm64] [--release noble]
#   Output: $VMIMG_DIR/<domain>-template.qcow2 (default ubuntu_vm-template.qcow2)
#
# What the template gets (cloud-init): user $VM_USER with the caller's pubkey,
# passwordless sudo, qemu-guest-agent, openssh — i.e. exactly the surface
# oneshot.sh's bootstrap needs. MIRA itself is NOT installed here; oneshot
# deploys it fresh every time.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/lib.sh"
ARCH="${ARCH:-$(uname -m)}"
case "$ARCH" in x86_64) ARCH=amd64 ;; aarch64|arm64) ARCH=arm64 ;; esac
RELEASE="${RELEASE:-noble}"
CLOUD_IMAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cloud-image) shift; CLOUD_IMAGE="${1:?}" ;;
    --arch) shift; ARCH="${1:?}" ;;
    --release) shift; RELEASE="${1:?}" ;;
    --*) f="${1#--}"; case "$f" in
      host|domain|vm-user|snap-dir|vmimg-dir|template|xml|pubkey|ip|vm-pass) shift; set_common "$f" "${1:?--$f needs a value}" ;;
      *) echo "unknown flag --$f" >&2; exit 2 ;; esac ;;
  esac
  shift
done
OUT="$VMIMG_DIR/$DOMAIN-template.qcow2"
SEED="$VMIMG_DIR/$DOMAIN-seed.iso"
WORK="$VMIMG_DIR/$DOMAIN-basebuild.qcow2"
[ -f "$OUT" ] && { echo "FATAL: $OUT already exists" >&2; exit 1; }
[ "$IS_LIBVIRT" = 1 ] && [ -z "$REMOTE_HOST" ] \
  || { echo "FATAL: run make-base ON the libvirt machine (local mode)" >&2; exit 1; }
finish_flags
PUB=$(pubkey)

if [ -z "$CLOUD_IMAGE" ]; then
  CLOUD_IMAGE="$VMIMG_DIR/ubuntu-$RELEASE-server-cloudimg-$ARCH.img"
  [ -f "$CLOUD_IMAGE" ] || {
    URL="https://cloud-images.ubuntu.com/$RELEASE/current/$RELEASE-server-cloudimg-$ARCH.img"
    echo "downloading $URL"
    wget -O "$CLOUD_IMAGE" "$URL"
  }
fi
echo "== build disk (20G) + seed =="
qemu-img convert -O qcow2 "$CLOUD_IMAGE" "$WORK"
qemu-img resize "$WORK" 20G
USERDATA=$(mktemp); META=$(mktemp)
{ echo "#cloud-config"
  echo "hostname: mira-base"
  echo "ssh_pwauth: false"
  echo "users:"
  echo "  - name: $VM_USER"
  echo "    groups: sudo"
  echo "    shell: /bin/bash"
  echo "    sudo: ALL=(ALL) NOPASSWD:ALL"
  echo "    lock_passwd: true"
  echo "    ssh_authorized_keys:"
  echo "      - $PUB"
  echo "packages: [qemu-guest-agent, openssh-server]"
  echo "runcmd: [systemctl enable --now qemu-guest-agent]"
} > "$USERDATA"
echo "instance-id: mira-base-$(date +%s)" > "$META"
if command -v cloud-localds >/dev/null; then
  cloud-localds "$SEED" "$USERDATA" "$META"
else
  ISOIMG=$(mktemp -d); mkdir -p "$ISOIMG/cidata"
  cp "$USERDATA" "$ISOIMG/cidata/user-data"; cp "$META" "$ISOIMG/cidata/meta-data"
  ( genisoimage -output "$SEED" -volid cidata -joliet -rock "$ISOIMG/cidata" \
    || mkisofs -output "$SEED" -volid cidata -joliet -rock "$ISOIMG/cidata" \
    || xorriso -as mkisofs -output "$SEED" -volid cidata -joliet -rock "$ISOIMG/cidata" )
  rm -rf "$ISOIMG"
fi

echo "== boot builder domain =="
sed -e "s|__DOMAIN__|$DOMAIN-basebuild|g" -e "s|__DISK__|$WORK|" "$VM_XML" > /tmp/basebuild.xml
# Attach the cloud-init seed as a read-only sata cdrom: Ubuntu cloud images
# have no path-based datasource, so the seed must be a domain device for
# cloud-init to consume it. base-vm.xml stays the shared template skeleton
# (oneshot.sh renders it with __DOMAIN__/__DISK__ only), so the builder's
# cdrom is injected into the rendered XML here, not into the skeleton.
SEEDFRAG=$(mktemp)
printf '%s\n' \
  "    <disk type='file' device='cdrom'>" \
  "      <driver name='qemu' type='raw'/>" \
  "      <source file='$SEED'/>" \
  "      <target dev='sda' bus='sata'/>" \
  "      <readonly/>" \
  "    </disk>" > "$SEEDFRAG"
sed "/^    <\/disk>$/r $SEEDFRAG" /tmp/basebuild.xml > /tmp/basebuild.xml.cd
mv /tmp/basebuild.xml.cd /tmp/basebuild.xml
rm -f "$SEEDFRAG"
VIRSH define /tmp/basebuild.xml
VIRSH start "$DOMAIN-basebuild"
# wait_guest_agent/vmexec bind to the file-global $DOMAIN (lib.sh) — the live
# VM's name, not the builder's. Scope DOMAIN to the started builder for the
# boot/verify block so the readiness probes target $DOMAIN-basebuild, and
# restore the caller's domain right after (the shutdown block and $OUT/$SEED
# paths use the original $DOMAIN).
SAVED_DOMAIN=$DOMAIN
DOMAIN=$DOMAIN-basebuild
wait_guest_agent
vmexec "cloud-init status --wait >/dev/null 2>&1; echo CLOUDINIT-OK"
vmexec "id $VM_USER >/dev/null && sudo -n true && echo BASE-READY"
DOMAIN=$SAVED_DOMAIN

echo "== shutdown + freeze =="
VIRSH shutdown "$DOMAIN-basebuild"
for _ in $(seq 1 60); do [ "$(VIRSH domstate "$DOMAIN-basebuild")" = "shut off" ] && break; sleep 2; done
[ "$(VIRSH domstate "$DOMAIN-basebuild")" = "shut off" ] \
  || { echo "FATAL: shutdown timed out" >&2; exit 1; }
VIRSH undefine "$DOMAIN-basebuild"
mv "$WORK" "$OUT"
rm -f "$SEED" /tmp/basebuild.xml
echo "== DONE: base template at $OUT — oneshot.sh can now spawn from it =="
