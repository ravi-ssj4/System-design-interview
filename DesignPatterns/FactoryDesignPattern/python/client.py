from util import VehicleFactory
from util import Car
from util import Bike

def inefficientClient(vehicleType):
    vehicle = None
    if vehicleType == "Car":
        vehicle = Car()
    elif vehicleType == "Bike":
        vehicle = Bike()
    if vehicle != None:
        vehicle.createVehicle()
    else:
        print("Error: Unknown vehicle type")

def efficientClient(vehicleType):
    vehicle = VehicleFactory.getVehicle(vehicleType)
    if vehicle is not None:
        vehicle.createVehicle()
    else:
        print("Error: Unknown vehicle type")


if __name__ == "__main__":
    vehicleType = input('Enter the vehicle Type: ')
    inefficientClient(vehicleType)
    efficientClient(vehicleType)
