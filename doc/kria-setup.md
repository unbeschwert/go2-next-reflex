# Kria KV260 setup

Condensed from the RIVER SDK cheatsheet "Get started with Kria".

## C++ development and deployment

1. **Laptop:** edit the ROS 2 C++ code.
2. **Laptop:** cross-compile for the Kria using the Docker builder in `data/go2-kria`. [Build steps](go2-kria-overview.md#docker-build-and-deployment)
3. **Laptop → Kria:** copy the build over SSH.
4. **Kria:** start the ROS 2 application.
5. **Kria → Go2:** the application talks to the robot over an Ethernet cable.
6. **Go2:** the robot's own controller carries out the commands.

The five-output Python workflow installs the same overlay and copies the tracked
scripts to `/root/`. [Python bridge setup](gesture-pipeline.md#board-setup-once-per-sd-card)

The Kria has only **one Ethernet port**, so use the supplied **USB-to-Ethernet adapter** for the robot:
- Kria's built-in port → laptop
- USB adapter → Go2

The organisers tried it the other way round and the laptop's firewall blocked it, so don't bother.

Without a second cable you can use the UART console, and copy files by moving the SD card to your laptop. That gets annoying fast.

## The board

| Side | What's there |
|---|---|
| 1 | SD card slot, camera ribbon-cable slot (next to the SD slot), UART port |
| 2 | RESET button |
| 3 (opposite side 1) | Power, display outputs, 4 USB ports, Ethernet |

## Connecting everything

1. Insert the SD card.
2. Attach the camera.
3. *(Optional)* UART to the laptop.
4. *(Optional)* A screen over HDMI.
5. *(For the demo)* The USB-to-Ethernet adapter, to the robot.
6. *(Optional)* The Ethernet port, to the laptop.
7. Power on last.

Log in as `root`; there's no password:

```bash
ssh root@durance-kria.local
# or over UART (the port may be /dev/ttyUSB0–3)
sudo minicom -D /dev/ttyUSB1 -b 115200
```

On UART you see the boot logs straight away. Over Ethernet, wait for it to finish booting.

## Health checks

```bash
lsusb      # expect at least 4 lines (two root hubs + two Microchip hubs); only 2 → SD card probably faulty
ifconfig   # expect eth0 (e.g. 192.168.77.1) and lo with addresses
```

## Network to the Go2

```bash
ip -br link
# lo        UNKNOWN
# eth0      UP      ← built-in port, to the laptop
# enu1u1c2  UP      ← USB adapter, to the Go2 (name depends on the USB port: enu1u1c2 … enu1u4c2)
```

- `/etc/systemd/network/10-unitree-go2.network` sets up the robot connection automatically. It's tied to **the adapter's MAC address**: if your adapter has a different MAC, edit that file and reboot.
- The Go2 is at `192.168.123.161`. The Kria's side should be `192.168.123.99/24`.

Test it:

```bash
ip -br addr                              # 1. both interfaces have addresses
ip route get 192.168.123.161             # 2. the route to the Go2 uses the adapter
ping -I enu1u4c2 -c 4 192.168.123.161    # 3. the robot answers (use your adapter's name)
ip route get 192.168.77.2                # 4. the laptop still goes via eth0 (use your laptop's address)
source /opt/ros/humble/setup.bash        # 5. with CycloneDDS configured (see go2-kria-overview.md):
ros2 topic list -t
```

## Running the FPGA side

### SPLEAT server (runs the SNN on the FPGA)

It normally starts at boot:

```bash
systemctl status spleat-server.service
journalctl -u spleat-server.service      # logs
```

To start it by hand, with logs in the terminal:

```bash
systemctl stop spleat-server.service && xmutil loadapp spleat && backward-server -c /root/.config/spleat-server/config.yaml
```

### Despot (camera → frames → SPLEAT)

With some board/camera combinations the first launch after boot freezes. **Run this once after every reboot** to avoid it:

```bash
despot_test_camera      # takes a few seconds; stuck >10 s → the board may be faulty
```

Then start the camera app:

```bash
despot_send_and_forget --to_display 0                                     # no screen
DISPLAY=:0 SDL_VIDEODRIVER=x11 despot_send_and_forget --to_display 1      # with a screen
```

Its settings live in `/root/metavision_camera.json`; see [deploy-model.md](deploy-model.md).
