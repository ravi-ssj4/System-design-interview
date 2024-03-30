#include "vehicle_factory.hpp"
#include <iostream>
using namespace std;

int main() {
    string vehicleType;
    cin>>vehicleType;
    Vehicle *vehicle = VehicleFactory::getVehicle(vehicleType);
    vehicle->createVehicle();
    return 0;
}

/*
    Pros of Factory:
        1. All the logic to build objects based on condition(veicleType here) is taken care by the factory and not the client
        2. the code therefore is loosly coupled
*/