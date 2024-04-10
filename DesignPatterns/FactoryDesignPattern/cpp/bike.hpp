#ifndef bike_hpp
#define bike_hpp

#include "Vehicle.hpp"

// Concrete Vehicle implementation -> Bike inherits from Vehicle and implements createVehicle()
class Bike: public Vehicle {
    public:
        void createVehicle();
};

#endif