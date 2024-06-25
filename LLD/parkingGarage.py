import heapq
from typing import List

class ParkingSpot:
    def __init__(self, name: str, size: int, distanceList: List[int]) -> None:
        self.name = name
        self.size = size
        self.distanceList = distanceList

class ParkingGarage:
    def __init__(self, parkingSpots: List[ParkingSpot], num_entrances: int) -> None:
        self.carMap = {} # to map the car license plate no with the parking spot
        self.priorityQueues = [[[] for _ in range(3)] for _ in range(num_entrances)] # a multidimensional priority queue: (num_entrances * size) priority queues
        self.spotToPriorityQueuesLocation = {} # to store a mapping of parking spot name: position of this spot in each of its priority queues
        self.num_entrances = num_entrances
        self.parkingSpots = parkingSpots

        for entranceNo in range(self.num_entrances):
            for parkingSpot in self.parkingSpots:
                size = parkingSpot.size
                distanceFromEntrance = parkingSpot.distanceList[entranceNo]
                heapq.heappush(self.priorityQueues[entranceNo][size], [distanceFromEntrance, parkingSpot])
        
        for entranceNo in range(self.num_entrances):
            for size in range(3):
                pQueue = self.priorityQueues[entranceNo][size]
                
                parkingSpotPositions = []
                for i in range(len(pQueue)):
                    _,  parkingSpot = pQueue[i]
                    parkingSpotPositions.append([entranceNo, size, i]) # entranceno: row, size: col, i: position in the priority queue list
                    self.spotToPriorityQueuesLocation[parkingSpot.name] = parkingSpotPositions

    # -> each size has about 5 priority queues 
    # -> all of them will contain same number of elements(spots) 
    # -> size of one is enough -> hence iterating through size
    def getNumSpots(self):
        numSpots = 0
        for s in range(len(self.priorityQueues[0])):
            numSpots += len(self.priorityQueues[0][s])
        return numSpots
    
    def entry(self, licenceNo, size, entranceNo):
        # adding the mapping of licenceNo -> parkingSpot.name in the car map
        # assignment of parking spot -> removal of spot from all the priority queues
        # followed by removal of the spot from the hashmap as well
        distanceFromSpot, bestAvailableParkingSpot = self.priorityQueues[entranceNo][size][0] # first element of the pQueue
        print(distanceFromSpot, bestAvailableParkingSpot.name)

        self.carMap[licenceNo] = bestAvailableParkingSpot.name

        positions = self.spotToPriorityQueuesLocation[bestAvailableParkingSpot.name]

        print(positions)
        for i, j, idxInPQ in positions:
            print(self.priorityQueues[i][j][idxInPQ])
        
        del self.spotToPriorityQueuesLocation[bestAvailableParkingSpot.name]
    
    def exit(self, licenceNo, size, entranceNo):
        previouslyOccupiedSpot: ParkingSpot = self.carMap[licenceNo]
        # add this spot back to all the priority queues
        for entranceNum in range(self.num_entrances):
            distFromEntrance = previouslyOccupiedSpot.distanceList[entranceNum]
            heapq.heappush(self.priorityQueues[entranceNum][size], [distFromEntrance, previouslyOccupiedSpot])
        
        # add this spot to the position hash map as well -> same logic as in the constructor
        for entranceNum in range(self.num_entrances):

            pQueues = self.priorityQueues[entranceNum][size]
                
            parkingSpotPositions = []
            for i in range(len(pQueues)):
                _,  parkingSpot = pQueues[i]
                parkingSpotPositions.append([entranceNo, size, i]) # entranceno: row, size: col, i: position in the priority queue list
                self.spotToPriorityQueuesLocation[parkingSpot.name] = parkingSpotPositions

            self.spotToPriorityQueuesLocation[previouslyOccupiedSpot.name] = parkingSpotPositions






def main():
    parkingSpot1 = ParkingSpot("x1", 0, [10, 20, 30, 40, 50])
    parkingSpot2 = ParkingSpot("x2", 0, [100, 200, 300, 400, 500])

    parkingGarage = ParkingGarage([parkingSpot1, parkingSpot2], 5)
    
    for e in range(5):
        for s in range(3):
            pQueues = parkingGarage.priorityQueues[e][s]
            for dist, parkingSpot in pQueues:
                print(dist, parkingSpot.name)

    print(parkingGarage.getNumSpots())

    parkingGarage.entry("AXAPR", 0, 0)

    for e in range(5):
        for s in range(3):
            pQueues = parkingGarage.priorityQueues[e][s]
            for dist, parkingSpot in pQueues:
                print(dist, parkingSpot.name)

    print(parkingGarage.getNumSpots())

main()
