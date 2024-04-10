#ifndef car_hpp
#define car_hpp

#include "vehicle.hpp"

// Concrete Vehicle implementation -> Car inherits from Vehicle and implements createVehicle()
class Car: public Vehicle {
    public:
        void createVehicle();
};

#endif