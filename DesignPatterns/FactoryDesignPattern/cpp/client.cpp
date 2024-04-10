#include <iostream>
#include "car.hpp"
#include "bike.hpp"
using namespace std;

int main() {
    string vehicleType;
    cin>>vehicleType;
    
    Vehicle *vehicle;
    
    if (vehicleType == "Car") {
        vehicle = new Car();
    } else if (vehicleType == "Bike") {
        vehicle = new Bike();
    } else {
        cout << "Error" << endl;
        return -1;
    }
    vehicle->createVehicle();
    return 0;
}

/* 
    Issue: 
        1. Everytime a new Vehicle is added to the Vehicle inventory, the client has to make changes in its code 
        2. Code is tightly coupled therefore to the client code
    
    Solution: Take this object creation logic(line 12 - 19) based on condition into a separate location : A Factory! -> VehicleFactory
*/
