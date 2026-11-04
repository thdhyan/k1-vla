# Booster K1 fleet (as of 2026-10-27)

Machine-readable copy: `../src/k1_chat/fleet.json` (loader `k1_chat.fleet`). Keep both in sync.

| Robot | IP (static, robotics_5G) | Firmware | Status |
|---|---|---|---|
| A1 | none | - | out of commission |
| A2 | 192.168.50.215 | **1.8.1.2-release-02884** (updated 2026-10-27) | active |
| A3 | 192.168.50.230 | 1.6.2.2-release.global-02145 | active |
| B1 | 192.168.50.105 | 1.6.2.2 | active |
| B2 | 192.168.50.62 | last record 1.7.0.7 | identity unconfirmed |
| B3 | 192.168.50.248 | 1.6.2.2 | active |

Nicknames in `/opt/booster/robot_info.txt`: `Booster K1_UMN_<name>`. Hardware: K1 Orin NX, aarch64, 22 DoF, 4 cores, ~7.4 GB RAM, ~390 GB disk, Ubuntu 22.04, ROS 2 Humble, JetPack 6.

## Network
- Lab SSID `robotics_5G` (underscore; 2.4 GHz `robotics`), password: ask the lab (not kept in git). Subnet 192.168.50.0/24, router 192.168.50.1 (no internet of its own).
- Laptop `enp2s0` -> router: 192.168.50.250 (robots' gateway) and 192.168.123.222 (Unitree G1 subnet). Laptop `wlo1` is on eduroam.
- Robot internet = laptop NAT (not persistent): `sysctl net.ipv4.ip_forward=1` + nft table `ip k1nat` (masquerade out `wlo1`, saddr 192.168.50.0/24). Undo: `sudo nft delete table ip k1nat`. Robots use gateway .250, DNS 8.8.8.8/1.1.1.1.
- Robot internal NICs (all K1): `usb_eth0` 192.168.127.101 (body<->compute), `enP8p1s0` 192.168.13.101, wired docs address 192.168.10.102 (host 192.168.10.10), `wlP1p1s0` wifi.
- Older network "Hello" (iPhone hotspot, dynamic IPs) is not used any more.

## Services seen on A2 (fw 1.8.1.2)
`:22` ssh, `:50051` RPC, `:50052` update-engine, `:51111/51112/51113` Studio websockets, `:16888/16889/6868` agent manager, `:20011` containerd gateway, UDP `:9000` heartbeat broadcast.
Heartbeat fields: serial, model, firmware, IP, nickname. Dashboard in `booster_ws/tools/fleet_dashboard` reads it.

## Hardware relevant to chat
- Audio: USB mic array "NationalChip UAC Speaker" (capture, has direction-of-arrival), USB audio playback device. ROS `/booster/audio/*` and `/LuiApiTopicReq|Resp`.
- Camera: head stereo pair. `/boostercamera/head/rgb` (nv12 544x448, ~27 Hz on 1.8.1.2), `/boostercamera/head/depth` (float32 **mm**, ~25 Hz). On 1.6.2.2 only ~4 Hz.
