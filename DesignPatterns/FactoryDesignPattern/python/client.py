from util import VehicleFactory

if __name__ == "__main__":
    vehicleType = input('Enter the vehicle Type: ')
    vehicle = VehicleFactory.getVehicle(vehicleType)
    if vehicle is not None:
        vehicle.createVehicle()
    else:
        print("Error: Unknown vehicle type")
