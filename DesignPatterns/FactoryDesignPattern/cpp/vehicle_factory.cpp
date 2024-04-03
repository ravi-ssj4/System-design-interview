#include "vehicle_factory.hpp"

Vehicle * VehicleFactory::getVehicle(string vehicleType) {
    Vehicle *vehicle;
    if (vehicleType == "Car") {
        vehicle = new Car();
    } else if (vehicleType == "Bike") {
        vehicle = new Bike();
    } else {
        cout << "Error" << endl;
        return NULL;
    }
    return vehicle;
}