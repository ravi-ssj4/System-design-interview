#ifndef vehicle_factory_hpp
#define vehicle_factory_hpp

#include <iostream>
#include "car.hpp" // also includes vehicle.hpp
#include "bike.hpp" // also includes vehicle.hpp -> not included again due to the ifndef checks
using namespace std;

class VehicleFactory {
    public:
        static Vehicle* getVehicle(string vehicleType); // static function ensures we can directly use it without creating object of VehicleFactory(as its not needed)
};

#endif