from abc import ABC, abstractmethod

# Abstract class vehicle (inherits ABC - Abstract Base Class)
class Vehicle(ABC):
    @abstractmethod
    def createVehicle(self):
        pass

# Concrete Car Class -> implements Vehicle
class Car(Vehicle):
    def createVehicle(self):
        print("Car has been created")

# Concrete Bike Class -> implements Vehicle
class Bike(Vehicle):
    def createVehicle(self):
        print("Bike has been created")

# Vehicle Factory
class VehicleFactory:
    @staticmethod
    def getVehicle(vehicleType):
        if vehicleType == "Car":
            return Car()
        elif vehicleType == "Bike":
            return Bike()
        else:
            return None
