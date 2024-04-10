// to avoid multiple inclusions of header files
#ifndef vehicle_hpp
#define vehicle_hpp

// Abstract Class Vehicle
class Vehicle { 
    public:
        virtual void createVehicle() = 0; // pure virtual function -> assigning to 0 makes it compulsary for the base class to implement this function
};

#endif