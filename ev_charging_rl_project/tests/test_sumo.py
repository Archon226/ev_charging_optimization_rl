import os, sys

# Check SUMO_HOME
sumo_home = os.environ.get("SUMO_HOME")
print("SUMO_HOME =", sumo_home)
assert sumo_home, "SUMO_HOME is not set"

# Add SUMO tools to path
tools = os.path.join(sumo_home, "tools")
sys.path.append(tools)

# Import SUMO Python libs
import sumolib, traci

print("sumolib imported from", sumolib.__file__)
print("traci imported from", traci.__file__)
print("OK")
