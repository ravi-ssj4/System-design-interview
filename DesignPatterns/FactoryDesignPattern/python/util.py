from abc import ABC, abstractmethod

# Abstract class vehicle
class Vehicle:
    @abstractmethod
    def createVehicle(self):
        pass

# Concrete Car Class
class Car(Vehicle):
    def createVehicle(self):
        print("Car has been created")

# Concrete Bike Class
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
