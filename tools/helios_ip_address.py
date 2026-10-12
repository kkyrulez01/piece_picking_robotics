from arena_api.system import system

def ip_to_int(ip_str):
    octets = [int(x) for x in ip_str.split(".")]
    return (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]

devices = system.create_device()
if not devices:
    print("Camera not found!")
    exit(1)

cam = devices[0]
nm = cam.nodemap

model_name = nm['DeviceModelName'].value
serial_num = nm['DeviceSerialNumber'].value
print(f"Configuring {model_name} ({serial_num})...")

# 1. Store persistent network parameters
nm["GevPersistentIPAddress"].value = ip_to_int("192.168.1.130")
nm["GevPersistentSubnetMask"].value = ip_to_int("255.255.255.0")
nm["GevPersistentDefaultGateway"].value = ip_to_int("192.168.1.1")

# # 2. Tell the camera to boot into Persistent IP mode (DHCP disabled)
# (LLA is hardwired as True by the GigE Vision spec, so it stays active automatically)
nm["GevCurrentIPConfigurationPersistentIP"].value = True  # Prioritize static IP
nm["GevCurrentIPConfigurationDHCP"].value = False          # Skip DHCP search

print("Configured successfully!")
print("Persistent IP: 192.168.1.130 | Subnet: 255.255.255.0 | Gateway: 192.168.1.1")
print("Flags: PersistentIP=ON, DHCP=OFF, LLA=ON (Safety Fallback)")

print("Rebooting camera to apply settings...")

# 3. Trigger camera soft reset to boot with new IP
nm["DeviceReset"].execute()

system.destroy_device()
print("Done! Allow ~15 seconds for the camera to finish booting.")

system.destroy_device()